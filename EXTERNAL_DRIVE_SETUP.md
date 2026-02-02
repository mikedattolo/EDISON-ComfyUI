# External Drive Setup for Large Models

This guide walks you through mounting a large external drive (2TB or more) to store very large language models that won't fit on your main system drive.

## Overview

EDISON supports storing models on external drives, allowing you to use massive models (70B+ parameters) without filling up your main system storage. This is configured through the `large_model_path` option in the configuration file.

## Prerequisites

- External drive (2TB+ recommended for large models)
- Drive formatted as NTFS, ext4, or exFAT
- Root/sudo access to your Linux system

## Step 1: Identify Your Drive

First, connect your external drive and identify it:

```bash
# List all block devices
lsblk

# Identify your drive (look for the size and name, e.g., sdb1)
sudo fdisk -l
```

You should see output like:
```
/dev/sdb1    2048  3907028991  3907026944  1.8T  7 HPFS/NTFS/exFAT
```

Note the device path (e.g., `/dev/sdb1`) and UUID for the next steps.

## Step 2: Get Drive UUID

For permanent mounting, we need the drive's UUID:

```bash
# Get the UUID
sudo blkid /dev/sdb1

# Output will look like:
# /dev/sdb1: UUID="3C2A54FA2A54B29E" TYPE="ntfs" ...
```

Copy the UUID value (e.g., `3C2A54FA2A54B29E`).

## Step 3: Create Mount Point

Create a directory where the drive will be mounted:

```bash
sudo mkdir -p /mnt/models
sudo chown $USER:$USER /mnt/models
```

## Step 4: Install Required Packages

### For NTFS drives (Windows formatted):

```bash
sudo apt-get update
sudo apt-get install ntfs-3g
```

### For exFAT drives:

```bash
sudo apt-get update
sudo apt-get install exfat-fuse exfat-utils
```

### For ext4 drives (Linux formatted):

No additional packages needed - already supported by Linux.

## Step 5: Handle Windows Hibernation (NTFS only)

If your NTFS drive was previously used on Windows and has hibernation data, you'll need to clear it:

```bash
# Fix hibernation
sudo ntfsfix /dev/sdb1

# If that doesn't work, force it:
sudo ntfsfix -d /dev/sdb1
```

## Step 6: Mount the Drive

### Option A: Manual Mount (Temporary)

For NTFS:
```bash
sudo mount -t ntfs-3g -o uid=$(id -u),gid=$(id -g),umask=022 /dev/sdb1 /mnt/models
```

For exFAT:
```bash
sudo mount -t exfat -o uid=$(id -u),gid=$(id -g) /dev/sdb1 /mnt/models
```

For ext4:
```bash
sudo mount /dev/sdb1 /mnt/models
sudo chown -R $USER:$USER /mnt/models
```

### Option B: Automatic Mount (Permanent)

Edit `/etc/fstab` to mount automatically on boot:

```bash
sudo nano /etc/fstab
```

Add one of these lines (replace UUID with your drive's UUID):

**For NTFS:**
```
UUID=3C2A54FA2A54B29E /mnt/models ntfs-3g uid=1000,gid=1000,umask=022,defaults 0 0
```

**For exFAT:**
```
UUID=3C2A54FA2A54B29E /mnt/models exfat uid=1000,gid=1000,defaults 0 0
```

**For ext4:**
```
UUID=3C2A54FA2A54B29E /mnt/models ext4 defaults 0 2
```

Save and test the mount:
```bash
sudo mount -a
```

## Step 7: Create Model Directory

Create the directory structure for LLM models:

```bash
mkdir -p /mnt/models/llm
```

## Step 8: Configure EDISON

Edit the EDISON configuration file:

```bash
nano /workspaces/EDISON-ComfyUI/config/edison.yaml
```

Add or update the `large_model_path` setting:

```yaml
# Large model storage (external drive)
large_model_path: "/mnt/models/llm"
```

Save the file.

## Step 9: Verify Setup

Check that the drive is mounted and accessible:

```bash
# Check mount
df -h | grep /mnt/models

# Check permissions
ls -la /mnt/models

# Test write access
touch /mnt/models/test.txt && rm /mnt/models/test.txt && echo "✅ Write access OK"
```

## Step 10: Download Models

Now you can download large models to the external drive:

```bash
# Example: Download Qwen2.5-72B-Instruct
cd /mnt/models/llm
wget https://huggingface.co/Qwen/Qwen2.5-72B-Instruct-GGUF/resolve/main/qwen2.5-72b-instruct-q4_k_m.gguf
```

EDISON will automatically scan both `/opt/edison/models/llm` and `/mnt/models/llm` for available models.

## Verification

Start or restart EDISON and check the logs:

```bash
# View model scanning logs
journalctl -u edison -f | grep -i "scanning"
```

You should see messages indicating both model directories are being scanned:
```
INFO: Scanning for models in: /opt/edison/models/llm
INFO: Scanning for models in: /mnt/models/llm
INFO: Found 5 models total
```

## Troubleshooting

### Drive not mounting

**Error:** `mount: wrong fs type, bad option, bad superblock...`

**Solution:** Make sure you've installed the correct filesystem driver (ntfs-3g, exfat-fuse, etc.)

### Permission denied errors

**Error:** Cannot write to `/mnt/models`

**Solution:** Fix ownership:
```bash
sudo chown -R $USER:$USER /mnt/models
```

### Windows hibernation issues

**Error:** `The disk contains an unclean file system (0, 0)`

**Solution:** Clear hibernation data:
```bash
sudo ntfsfix -d /dev/sdb1
```

### Drive not auto-mounting on boot

**Error:** Drive not mounted after reboot

**Solution:** Check `/etc/fstab` for typos and test:
```bash
sudo mount -a
```

### Out of space

**Error:** `No space left on device`

**Solution:** Check available space:
```bash
df -h /mnt/models
```

Large models can be 40-90GB each. Make sure you have enough space.

## Model Size Reference

Here are approximate sizes for common quantization levels:

| Model | Q4_K_M | Q5_K_M | Q6_K | Q8_0 |
|-------|--------|--------|------|------|
| Qwen2.5-14B | ~8GB | ~10GB | ~12GB | ~15GB |
| Qwen2.5-32B | ~20GB | ~24GB | ~28GB | ~35GB |
| Qwen2.5-72B | ~44GB | ~53GB | ~62GB | ~77GB |
| DeepSeek V3 | ~90GB | ~110GB | ~130GB | ~160GB |

**Recommendation:** Use Q4_K_M for the best balance of quality and size.

## Unmounting Safely

Before disconnecting the external drive:

```bash
# Sync and unmount
sync
sudo umount /mnt/models
```

**Important:** Never disconnect the drive while EDISON is running and using models from it. Always stop the service first:

```bash
sudo systemctl stop edison
sudo umount /mnt/models
# Now safe to disconnect drive
```

## Best Practices

1. **Use Q4_K_M quantization** - Best balance of quality and file size
2. **Keep frequently used models on main drive** - Faster access times
3. **Store rarely used/experimental models on external drive** - Save main drive space
4. **Regular backups** - External drives can fail; keep backups of important models
5. **Fast drive recommended** - Use USB 3.1+ or Thunderbolt for best performance
6. **Monitor space** - Large models add up quickly; keep an eye on available space

## Advanced: Multiple External Drives

You can configure multiple external drives by mounting them to different paths:

```bash
# Mount Drive 1
sudo mkdir -p /mnt/models1
sudo mount /dev/sdb1 /mnt/models1

# Mount Drive 2
sudo mkdir -p /mnt/models2
sudo mount /dev/sdc1 /mnt/models2
```

Then in `config/edison.yaml`, you can specify multiple paths (feature to be implemented):

```yaml
# Multiple large model paths
large_model_paths:
  - "/mnt/models1/llm"
  - "/mnt/models2/llm"
```

## Summary

You've successfully configured an external drive for large model storage! EDISON will now scan both your main model directory and the external drive, giving you access to much larger models without filling up your system drive.

Next steps:
- Download large models (see UPGRADE_TO_SOTA.md for recommendations)
- Test model selection in the UI
- Monitor performance (external drives may be slightly slower than internal SSDs)

## See Also

- [UPGRADE_TO_SOTA.md](UPGRADE_TO_SOTA.md) - Recommended large models
- [QUICK_DEPLOY.md](QUICK_DEPLOY.md) - Initial EDISON setup
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues and solutions
