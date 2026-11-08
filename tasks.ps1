<#
.SYNOPSIS
    Beacon task runner. The Windows-native stand-in for the Makefile, since
    there is no make here.

.DESCRIPTION
    Thin wrappers over uv and docker compose - no logic lives in this file.
    Every command it runs is also written out in README.md, so a Linux cloner
    can follow along without PowerShell.

.EXAMPLE
    .\tasks.ps1 up
    .\tasks.ps1 check
    .\tasks.ps1 migrate
#>

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('help', 'sync', 'up', 'down', 'ps', 'logs', 'lint', 'fmt', 'types',
                 'test', 'test-all', 'check', 'migrate', 'revision', 'gateway',
                 'retrieval', 'imports')]
    [string]$Task = 'help',

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

$Compose = @('compose', '-f', 'deploy/compose.yaml')

function Invoke-Checked {
    param([string]$Exe, [string[]]$Arguments)
    Write-Host "> $Exe $($Arguments -join ' ')" -ForegroundColor DarkCyan
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Exe exited with code $LASTEXITCODE"
    }
}

function Assert-Uv {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "uv is not on PATH. Install it with: irm https://astral.sh/uv/install.ps1 | iex  (then restart the shell)"
    }
}

function Assert-Docker {
    # Docker Desktop not running is the single most common local failure, and
    # its native error is unhelpful. Fail early and say what to do.
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw 'docker is not on PATH. Install Docker Desktop.'
    }
    & docker info *> $null
    if ($LASTEXITCODE -ne 0) {
        throw 'Docker Desktop is not running. Start it, wait for the whale icon to settle, then retry.'
    }
}

function Assert-ComposeFile {
    if (-not (Test-Path 'deploy/compose.yaml')) {
        throw 'deploy/compose.yaml does not exist yet. It is owned by agent A1 (Wave 1).'
    }
}

switch ($Task) {

    'help' {
        Write-Host ''
        Write-Host 'Beacon tasks' -ForegroundColor Cyan
        Write-Host '  sync       uv sync --locked          install the workspace from the lockfile'
        Write-Host '  imports    import every package      the Wave 0 acceptance check'
        Write-Host '  up         docker compose up -d      start postgres, redis, qdrant, redpanda, phoenix'
        Write-Host '  down       docker compose down       stop them (named volumes survive)'
        Write-Host '  ps         docker compose ps         should read healthy, not merely running'
        Write-Host '  logs       docker compose logs -f    follow; pass a service name to narrow'
        Write-Host '  lint       ruff check'
        Write-Host '  fmt        ruff format              rewrites files'
        Write-Host '  types      mypy'
        Write-Host '  test       pytest, unit only         integration tests need Docker'
        Write-Host '  test-all   pytest including integration'
        Write-Host '  check      lint + format --check + types + test   what CI runs'
        Write-Host '  migrate    alembic upgrade head'
        Write-Host '  revision   alembic revision --autogenerate -m "..."   ORCHESTRATOR ONLY'
        Write-Host '  gateway    run the gateway with reload'
        Write-Host '  retrieval  run the retrieval service with reload'
        Write-Host ''
    }

    'sync' { Assert-Uv; Invoke-Checked 'uv' @('sync', '--locked') }

    'imports' {
        Assert-Uv
        Invoke-Checked 'uv' @('run', 'python', '-c',
            'import beacon_core, beacon_llm, gateway, retrieval, match_worker, ingestion, evals; print("all workspace packages import")')
    }

    'up' {
        Assert-Docker; Assert-ComposeFile
        Invoke-Checked 'docker' ($Compose + @('up', '-d'))
        Invoke-Checked 'docker' ($Compose + @('ps'))
    }

    'down'  { Assert-Docker; Assert-ComposeFile; Invoke-Checked 'docker' ($Compose + @('down')) }
    'ps'    { Assert-Docker; Assert-ComposeFile; Invoke-Checked 'docker' ($Compose + @('ps')) }
    'logs'  { Assert-Docker; Assert-ComposeFile; Invoke-Checked 'docker' ($Compose + @('logs', '-f') + $Rest) }

    'lint'  { Assert-Uv; Invoke-Checked 'uv' @('run', 'ruff', 'check', '.') }
    'fmt'   { Assert-Uv; Invoke-Checked 'uv' @('run', 'ruff', 'format', '.') }
    'types' { Assert-Uv; Invoke-Checked 'uv' @('run', 'mypy', '.') }
    'test'  { Assert-Uv; Invoke-Checked 'uv' @('run', 'pytest', '-m', 'not integration') }

    'test-all' { Assert-Uv; Assert-Docker; Invoke-Checked 'uv' @('run', 'pytest') }

    'check' {
        Assert-Uv
        Invoke-Checked 'uv' @('run', 'ruff', 'check', '.')
        Invoke-Checked 'uv' @('run', 'ruff', 'format', '--check', '.')
        Invoke-Checked 'uv' @('run', 'mypy', '.')
        Invoke-Checked 'uv' @('run', 'pytest', '-m', 'not integration')
        Write-Host 'all green' -ForegroundColor Green
    }

    'migrate' { Assert-Uv; Assert-Docker; Invoke-Checked 'uv' @('run', 'alembic', 'upgrade', 'head') }

    'revision' {
        # Migrations are orchestrator-owned: two agents generating revisions in
        # parallel produce two heads and a manual merge.
        Assert-Uv; Assert-Docker
        if (-not $Rest) { throw 'Usage: .\tasks.ps1 revision "short message"' }
        Invoke-Checked 'uv' @('run', 'alembic', 'revision', '--autogenerate', '-m', ($Rest -join ' '))
    }

    'gateway' {
        Assert-Uv
        Invoke-Checked 'uv' @('run', 'uvicorn', 'gateway.main:app', '--reload', '--port', '8000')
    }

    'retrieval' {
        Assert-Uv
        Invoke-Checked 'uv' @('run', 'uvicorn', 'retrieval.main:app', '--reload', '--port', '8001')
    }
}
