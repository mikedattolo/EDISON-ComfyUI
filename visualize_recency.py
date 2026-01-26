#!/usr/bin/env python3
"""
Visualize recency boost decay over time
"""

import time

def visualize_recency_decay():
    """Show how recency boost decays over time"""
    print("=" * 70)
    print("Recency Boost Decay Visualization")
    print("=" * 70)
    print("\nFormula: recency_boost = max(0, min(1, 1 - (age_days / 30)))")
    print("Final Score: 0.85 * base_score + 0.15 * recency_boost\n")
    
    print("Age (days) | Recency Boost | Example Final Score (base=0.80)")
    print("-" * 70)
    
    base_score = 0.80
    
    test_ages = [0, 1, 3, 7, 15, 20, 25, 30, 45, 60, 90]
    
    for age in test_ages:
        # Calculate recency boost
        recency_boost = max(0, min(1, 1 - (age / 30)))
        
        # Calculate final score
        final_score = 0.85 * base_score + 0.15 * recency_boost
        
        # Create visual bar
        bar_length = int(recency_boost * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        
        print(f"{age:>3} days   | {recency_boost:>5.2f} [{bar}] | {final_score:.4f}")
    
    print("\n" + "=" * 70)
    print("Key Insights:")
    print("  • Full boost (1.0) for content from today")
    print("  • Linear decay: loses 1/30 boost per day")
    print("  • Half boost (0.5) at 15 days")
    print("  • Zero boost (0.0) at 30+ days")
    print("  • Maximum impact: +0.15 to final score")
    print("=" * 70)
    
    # Show comparison example
    print("\n" + "=" * 70)
    print("Example: Recent vs Old Memory")
    print("=" * 70)
    
    print("\nScenario: Two memories with similar base scores")
    print("\nMemory A: 'User prefers Python' (2 days old)")
    base_a = 0.88
    age_a = 2
    recency_a = max(0, min(1, 1 - (age_a / 30)))
    final_a = 0.85 * base_a + 0.15 * recency_a
    print(f"  Base Score:     {base_a:.2f}")
    print(f"  Recency Boost:  {recency_a:.2f}")
    print(f"  Final Score:    {final_a:.4f} ✅")
    
    print("\nMemory B: 'User prefers Java' (45 days old)")
    base_b = 0.91
    age_b = 45
    recency_b = max(0, min(1, 1 - (age_b / 30)))
    final_b = 0.85 * base_b + 0.15 * recency_b
    print(f"  Base Score:     {base_b:.2f}")
    print(f"  Recency Boost:  {recency_b:.2f}")
    print(f"  Final Score:    {final_b:.4f}")
    
    print(f"\nWinner: Memory A (recent) wins despite lower base score!")
    print(f"Margin: {final_a - final_b:+.4f}")
    print("=" * 70)


if __name__ == "__main__":
    visualize_recency_decay()
