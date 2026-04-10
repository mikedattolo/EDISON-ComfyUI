# Create a startup shortcut for EDISON Node Agent
# Run this in PowerShell as your user (not admin)

param(
    [Parameter(Mandatory=$true)]
    [string]$EdisonServer,

    [string]$NodeName = $env:COMPUTERNAME,
    [string]$Role = "cad",
    [string]$AgentPath = "$PSScriptRoot\edison_node_agent.py"
)

$StartupDir = [Environment]::GetFolderPath("Startup")
$ShortcutPath = Join-Path $StartupDir "EDISON-Node-Agent.lnk"

# Find Python
$PythonPath = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $PythonPath) {
    Write-Error "Python not found on PATH. Install Python 3.10+ first."
    exit 1
}

# Create shortcut
$WShell = New-Object -ComObject WScript.Shell
$Shortcut = $WShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = $PythonPath
$Shortcut.Arguments = "`"$AgentPath`" --server $EdisonServer --name `"$NodeName`" --role $Role"
$Shortcut.WorkingDirectory = Split-Path $AgentPath
$Shortcut.Description = "EDISON Node Agent"
$Shortcut.WindowStyle = 7  # Minimized
$Shortcut.Save()

Write-Host ""
Write-Host "Startup shortcut created at:" -ForegroundColor Green
Write-Host "  $ShortcutPath"
Write-Host ""
Write-Host "The EDISON Node Agent will start automatically at login."
Write-Host "To remove: delete the shortcut from $StartupDir"
