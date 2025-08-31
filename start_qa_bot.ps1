#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Start the RAG QA Bot for Generative AI Applications
    
.DESCRIPTION
    This script starts the QA Bot with automatic dependency installation
    and system checks. It's designed for Windows PowerShell.
    
.PARAMETER SkipInstall
    Skip dependency installation if specified
    
.EXAMPLE
    .\start_qa_bot.ps1
    
.EXAMPLE
    .\start_qa_bot.ps1 -SkipInstall
#>

param(
    [switch]$SkipInstall
)

# Set execution policy for current session
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "    RAG QA Bot - Generative AI Apps" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
Write-Host "🔍 Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ $pythonVersion" -ForegroundColor Green
    } else {
        throw "Python not found"
    }
} catch {
    Write-Host "❌ ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ and try again" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if pip is available
Write-Host "🔍 Checking pip installation..." -ForegroundColor Yellow
try {
    $pipVersion = pip --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ $pipVersion" -ForegroundColor Green
    } else {
        throw "pip not found"
    }
} catch {
    Write-Host "❌ ERROR: pip is not available" -ForegroundColor Red
    Write-Host "Please ensure pip is installed with Python" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Install dependencies if not skipped
if (-not $SkipInstall) {
    Write-Host ""
    Write-Host "📦 Installing/updating dependencies..." -ForegroundColor Yellow
    try {
        pip install -r requirements.txt
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Dependencies installed successfully" -ForegroundColor Green
        } else {
            Write-Host "⚠️  Some dependencies may have failed to install" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "⚠️  Dependency installation had issues, continuing anyway..." -ForegroundColor Yellow
    }
} else {
    Write-Host "⏭️  Skipping dependency installation" -ForegroundColor Yellow
}

# Check if main script exists
if (-not (Test-Path "main_qa_bot.py")) {
    Write-Host "❌ ERROR: main_qa_bot.py not found" -ForegroundColor Red
    Write-Host "Please ensure you're running this script from the project root directory" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "🚀 Starting QA Bot..." -ForegroundColor Green
Write-Host ""
Write-Host "The web interface will open in your browser at:" -ForegroundColor Cyan
Write-Host "http://localhost:7860" -ForegroundColor Cyan
Write-Host ""
Write-Host "To stop the bot, press Ctrl+C in this window" -ForegroundColor Yellow
Write-Host ""

# Start the QA Bot
try {
    python main_qa_bot.py
} catch {
    Write-Host ""
    Write-Host "❌ ERROR: Failed to start QA Bot" -ForegroundColor Red
    Write-Host "Error details: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "💡 Troubleshooting tips:" -ForegroundColor Yellow
    Write-Host "   1. Ensure all dependencies are installed: pip install -r requirements.txt"
    Write-Host "   2. Check if port 7860 is available"
    Write-Host "   3. Try running: python test_main.py"
    Write-Host "   4. Check the error messages above"
}

Write-Host ""
Write-Host "QA Bot stopped." -ForegroundColor Yellow
Read-Host "Press Enter to exit"
