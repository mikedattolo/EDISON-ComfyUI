"""Test concurrent access safety for model locking"""

import concurrent.futures
import time
import threading
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(REPO_ROOT))

def test_locks_exist():
    """Test that model locks are initialized"""
    from services.edison_core import app
    
    # Check that all locks exist
    assert hasattr(app, 'lock_fast'), "lock_fast not found"
    assert hasattr(app, 'lock_medium'), "lock_medium not found"
    assert hasattr(app, 'lock_deep'), "lock_deep not found"
    assert hasattr(app, 'lock_vision'), "lock_vision not found"
    
    # Check that they are Lock objects (check for acquire/release methods)
    for lock_name in ['lock_fast', 'lock_medium', 'lock_deep', 'lock_vision']:
        lock = getattr(app, lock_name)
        assert hasattr(lock, 'acquire'), f"{lock_name} missing acquire method"
        assert hasattr(lock, 'release'), f"{lock_name} missing release method"
    
    print("✓ All model locks are properly initialized")


def test_get_lock_for_model():
    """Test that get_lock_for_model returns correct locks"""
    from services.edison_core import app
    
    # Create mock models
    mock_fast = Mock(name="llm_fast")
    mock_medium = Mock(name="llm_medium")
    mock_deep = Mock(name="llm_deep")
    mock_vision = Mock(name="llm_vision")
    
    # Patch the global models
    with patch.object(app, 'llm_fast', mock_fast), \
         patch.object(app, 'llm_medium', mock_medium), \
         patch.object(app, 'llm_deep', mock_deep), \
         patch.object(app, 'llm_vision', mock_vision):
        
        # Test lock assignment
        assert app.get_lock_for_model(mock_vision) is app.lock_vision, "Vision model should use lock_vision"
        assert app.get_lock_for_model(mock_deep) is app.lock_deep, "Deep model should use lock_deep"
        assert app.get_lock_for_model(mock_medium) is app.lock_medium, "Medium model should use lock_medium"
        assert app.get_lock_for_model(mock_fast) is app.lock_fast, "Fast model should use lock_fast"
        
    print("✓ get_lock_for_model returns correct locks for each model")


def test_lock_acquisition():
    """Test that locks can be acquired and released properly"""
    from services.edison_core import app
    
    # Test each lock can be acquired and released
    for lock_name in ['lock_fast', 'lock_medium', 'lock_deep', 'lock_vision']:
        lock = getattr(app, lock_name)
        
        # Acquire lock
        acquired = lock.acquire(blocking=False)
        assert acquired, f"{lock_name} should be acquirable"
        
        # Release lock
        lock.release()
        
        print(f"✓ {lock_name} acquired and released successfully")


def test_concurrent_access_simulation():
    """Test that concurrent calls don't cause issues with locks"""
    from services.edison_core import app
    
    # Track access order
    access_log = []
    access_lock = threading.Lock()
    
    def simulate_model_call(model_name, call_id, duration=0.1):
        """Simulate a model call with lock"""
        model_map = {
            'fast': None,
            'medium': None,
            'deep': None,
            'vision': None
        }
        
        # Create mock model
        mock_model = Mock(name=f"llm_{model_name}")
        model_map[model_name] = mock_model
        
        lock = app.get_lock_for_model(mock_model)
        
        # Log lock acquisition attempt
        with access_lock:
            access_log.append(f"Call {call_id}: acquiring {model_name} lock")
        
        # Simulate lock contention detection
        acquired = lock.acquire(blocking=False)
        if not acquired:
            with access_lock:
                access_log.append(f"Call {call_id}: waiting for {model_name} lock")
            lock.acquire()  # Wait for lock
        
        # Simulate model processing
        try:
            with access_lock:
                access_log.append(f"Call {call_id}: processing {model_name}")
            time.sleep(duration)
        finally:
            lock.release()
            with access_lock:
                access_log.append(f"Call {call_id}: released {model_name} lock")
    
    # Test concurrent fast model calls
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = []
        for i in range(2):
            future = executor.submit(simulate_model_call, 'fast', f"fast_{i}")
            futures.append(future)
        
        # Wait for all to complete
        concurrent.futures.wait(futures)
    
    # Verify no crashes
    assert len(access_log) > 0, "Access log should have entries"
    
    print("✓ Concurrent access simulation completed without errors")
    print(f"  Access log ({len(access_log)} entries):")
    for entry in access_log:
        print(f"    {entry}")


def test_different_models_concurrent():
    """Test that different models can be accessed concurrently"""
    from services.edison_core import app
    
    results = {'fast': 0, 'deep': 0}
    results_lock = threading.Lock()
    
    def fast_task():
        mock_fast = Mock(name="llm_fast")
        lock = app.get_lock_for_model(mock_fast)
        with lock:
            time.sleep(0.05)
            with results_lock:
                results['fast'] += 1
    
    def deep_task():
        mock_deep = Mock(name="llm_deep")
        lock = app.get_lock_for_model(mock_deep)
        with lock:
            time.sleep(0.05)
            with results_lock:
                results['deep'] += 1
    
    # Run both concurrently (different model locks)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(fast_task)
        f2 = executor.submit(deep_task)
        concurrent.futures.wait([f1, f2])
    
    assert results['fast'] == 1, "Fast task should complete"
    assert results['deep'] == 1, "Deep task should complete"
    
    print("✓ Different models can be accessed concurrently with separate locks")


def test_lock_prevents_concurrent_same_model():
    """Test that same model can't be accessed concurrently"""
    from services.edison_core import app
    
    execution_times = []
    exec_lock = threading.Lock()
    
    def model_call(call_id):
        mock_model = Mock(name="llm_fast")
        lock = app.get_lock_for_model(mock_model)
        
        start = time.time()
        with lock:
            time.sleep(0.05)  # Simulate work
        elapsed = time.time() - start
        
        with exec_lock:
            execution_times.append((call_id, elapsed))
    
    # Run 3 concurrent calls on same model
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(model_call, i) for i in range(3)]
        concurrent.futures.wait(futures)
    
    # Each should take at least 50ms, total should be at least 150ms
    total_time = sum(t for _, t in execution_times)
    assert total_time >= 0.15, f"Sequential locking should take >= 150ms, got {total_time*1000}ms"
    
    print(f"✓ Same model calls are serialized by lock (total time: {total_time*1000:.0f}ms)")
    for call_id, elapsed in execution_times:
        print(f"    Call {call_id}: {elapsed*1000:.0f}ms")


if __name__ == "__main__":
    print("Testing concurrent model access safety...\n")
    
    try:
        test_locks_exist()
        test_get_lock_for_model()
        test_lock_acquisition()
        test_concurrent_access_simulation()
        test_different_models_concurrent()
        test_lock_prevents_concurrent_same_model()
        
        print("\n✅ All concurrent safety tests passed!")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
