# EDISON Node Agent — Setup Guide

Connect remote machines (like your engineering laptop) to your main EDISON AI server as worker nodes.

---

## Your Setup

| Machine | Specs | Role |
|---------|-------|------|
| **Main EDISON PC** | Your AI server | Hub / orchestrator |
| **Engineering Laptop** | i7, 64GB DDR4, RTX A3000 6GB, Win11 | CAD node (Rhino 7) |

---

## Quick Start (5 minutes)

### On your Engineering Laptop (Windows 11):

**1. Install Python 3.10+**
- Download from [python.org](https://python.org/downloads)
- **Check "Add Python to PATH"** during installation

**2. Get the agent files**

Copy the `tools/node-agent/` folder from your EDISON server to your laptop. You can:
- Use a USB drive
- Use network share: `\\YOUR_EDISON_IP\...`
- Or download from your EDISON git repo

**3. Run the installer**

Double-click `install_windows.bat` or open a terminal in the folder:

```cmd
cd C:\path\to\node-agent
install_windows.bat
```

It will:
- Verify Python is installed
- Install `requests` and `psutil`
- Install `pywin32` (for Rhino COM automation)
- Ask for your EDISON server IP
- Optionally start the agent

**4. Or run manually:**

```cmd
cd C:\path\to\node-agent
pip install -r requirements.txt
python edison_node_agent.py --server 192.168.1.100 --name "Engineering-Laptop" --role cad
```

Replace `192.168.1.100` with your EDISON server's actual IP address.

---

## Finding Your EDISON Server IP

On your main EDISON PC, run:
- **Linux/Mac:** `hostname -I` or `ip addr`
- **Windows:** `ipconfig`

Look for the `192.168.x.x` or `10.x.x.x` address on your local network.

---

## Rhino 7 Integration

For full Rhino 7 CAD control from EDISON:

1. Install `pywin32` on the laptop:
   ```cmd
   pip install pywin32
   ```

2. **Start Rhino 7 first**, then start the agent

3. The agent will connect to Rhino via COM automation and can:
   - Run Rhino commands
   - Execute Python scripts inside Rhino
   - Open Grasshopper definitions
   - Export files (3DM, STL, OBJ, etc.)
   - Open files in Rhino

### Example: Send a Rhino command from EDISON chat
Once connected, you can tell EDISON:
> "Run the MeshToNurb command on my CAD laptop"
> "Export the current Rhino file as STL on the engineering node"
> "Open the latest 3dm file on my CAD workstation"

---

## Auto-Start at Login

### Option A: Batch shortcut
1. Press `Win+R`, type `shell:startup`, press Enter
2. Create a shortcut pointing to:
   ```
   python "C:\path\to\node-agent\edison_node_agent.py" --server 192.168.1.100 --name "Engineering-Laptop" --role cad
   ```

### Option B: PowerShell script
```powershell
cd C:\path\to\node-agent
.\setup_autostart.ps1 -EdisonServer 192.168.1.100 -NodeName "Engineering-Laptop"
```

---

## Windows Firewall

The agent listens on port **9200** for direct commands and discovery. You may need to allow it:

```cmd
netsh advfirewall firewall add rule name="EDISON Node Agent" dir=in action=allow protocol=TCP localport=9200
```

Or allow it through Windows Security when prompted.

---

## Network Requirements

| Port | Direction | Purpose |
|------|-----------|---------|
| 9200 | Inbound on laptop | Agent HTTP API (commands, discovery) |
| 8811 | Outbound to EDISON | Registration, heartbeat, task polling |

Both machines must be on the same network (or have routable IPs).

---

## Managing Nodes

### Web UI
Visit `http://YOUR_EDISON_IP:8811/web/nodes.html` to see all connected nodes, their status, hardware, and capabilities.

### From EDISON Chat
- *"Show me my connected nodes"*
- *"What's the status of my CAD laptop?"*
- *"Send a Rhino command to the engineering node"*
- *"Ping my engineering laptop"*

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/nodes` | List all nodes |
| GET | `/nodes/{id}` | Get node details |
| POST | `/nodes/register` | Register/update a node |
| POST | `/nodes/{id}/heartbeat` | Node heartbeat |
| POST | `/nodes/{id}/command` | Send command to node |
| POST | `/nodes/{id}/task` | Submit task to node |
| POST | `/nodes/discover` | Scan LAN for agents |
| DELETE | `/nodes/{id}` | Remove a node |

---

## Troubleshooting

**Agent can't reach EDISON server**
- Check both machines are on the same network
- Verify the IP: `ping YOUR_EDISON_IP` from the laptop
- Make sure EDISON is running on port 8811

**Rhino COM not available**
- Start Rhino 7 before the agent
- Make sure `pywin32` is installed: `pip install pywin32`
- Rhino must be running (not just installed)

**Node shows as offline**
- The heartbeat interval is 30 seconds — wait a moment
- Check the agent terminal for error messages
- Nodes are marked offline after 120 seconds without heartbeat

**GPU not detected**
- Make sure NVIDIA drivers are installed
- Verify `nvidia-smi` works in cmd: `nvidia-smi`
- The RTX A3000 should be auto-detected

---

## Architecture

```
┌──────────────────────┐         ┌──────────────────────────────┐
│   EDISON AI Server   │◄────────│  Engineering Laptop (Win11)  │
│                      │  HTTP   │                              │
│  ┌────────────────┐  │ :8811   │  ┌─────────────────────────┐│
│  │  Node Manager   │  │◄───────│  │  EDISON Node Agent      ││
│  │  /nodes/* API   │  │        │  │  - Auto hardware detect  ││
│  └────────────────┘  │        │  │  - Heartbeat every 30s   ││
│                      │:9200──►│  │  - Task execution        ││
│  ┌────────────────┐  │        │  │  - Rhino 7 COM bridge    ││
│  │  Chat / AI      │  │        │  └─────────────────────────┘│
│  │  "send rhino    │  │        │                              │
│  │   cmd to node"  │  │        │  ┌─────────────────────────┐│
│  └────────────────┘  │        │  │  Rhino 7                 ││
│                      │        │  │  - NURBS modeling         ││
│  ┌────────────────┐  │        │  │  - Grasshopper            ││
│  │  Web UI         │  │        │  │  - Python scripting       ││
│  │  nodes.html     │  │        │  └─────────────────────────┘│
│  └────────────────┘  │        │                              │
└──────────────────────┘         └──────────────────────────────┘
```
