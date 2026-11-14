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
                 'retrieval', 'worker', 'dev', 'imports')]
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
        Write-Host '  worker     run the match worker      without it every match stays Queued'
        Write-Host '  dev        start EVERYTHING          containers + 4 services in tabs, opens the console'
        Write-Host '             (same as double-clicking Start-Beacon.cmd; the worker bills only when you click Match)'
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

    'worker' {
        # The gateway only writes the request to the outbox. This process relays
        # it to Kafka AND consumes it, so without it a match never leaves Queued.
        # It calls Claude: running it spends API credits.
        #
        # beacon_llm defaults to REPLAY (recorded fixtures, never the network),
        # so that tests and any unconfigured process are free. The worker is
        # the one process that is meant to call Claude, so it opts into live
        # mode here - unless BEACON_LLM_MODE is already set, e.g. to "replay"
        # for a free dry run. Without this every real match failed at the
        # first node with FixtureMissingError.
        #
        # Cost: 96% of a match is the 20 rule checks. On Sonnet that is about
        # $0.35-0.43 per click; on Haiku about $0.10 (measured: two live runs, $0.103 and $0.107). The console is for trying
        # things, so it defaults to Haiku. Sonnet is for the one headline eval
        # run: set ANTHROPIC_DECISION_MODEL=claude-sonnet-5-5 first to use it.
        Assert-Uv; Assert-Docker
        if (-not $env:BEACON_LLM_MODE) { $env:BEACON_LLM_MODE = 'live' }
        if (-not $env:ANTHROPIC_DECISION_MODEL) { $env:ANTHROPIC_DECISION_MODEL = 'claude-haiku-5-5' }
        $perMatch = if ($env:ANTHROPIC_DECISION_MODEL -like '*sonnet*') { 'about $0.40' } else { 'about $0.10' }
        Write-Host "LLM mode: $($env:BEACON_LLM_MODE) · rule checks on $($env:ANTHROPIC_DECISION_MODEL) · $perMatch per Match trials click" -ForegroundColor Yellow
        Invoke-Checked 'uv' @('run', 'python', '-m', 'match_worker')
    }

    'dev' {
        # One command for the whole app. Each service still runs as its own
        # process with its own log - that is what lets one crash without the
        # others, and what makes the logs readable - but they open together:
        # as tabs of one Windows Terminal window when wt.exe exists, else as
        # separate PowerShell windows. Close a tab/window to stop that service.
        Assert-Uv; Assert-ComposeFile
        # One click should not need a second click on Docker Desktop first:
        # start it if its engine is not answering, and wait for it.
        & docker info *> $null
        if ($LASTEXITCODE -ne 0) {
            $exe = @(
                "$env:LOCALAPPDATA\Programs\DockerDesktop\Docker Desktop.exe",
                "$env:ProgramFiles\Docker\Docker\Docker Desktop.exe"
            ) | Where-Object { Test-Path $_ } | Select-Object -First 1
            if (-not $exe) { throw 'Docker Desktop is not running and was not found. Start it by hand, then retry.' }
            Write-Host '> starting Docker Desktop (first start can take a minute) ...' -ForegroundColor DarkCyan
            Start-Process -FilePath $exe
            for ($i = 0; $i -lt 60; $i++) {
                Start-Sleep -Seconds 3
                & docker info *> $null
                if ($LASTEXITCODE -eq 0) { break }
            }
        }
        Assert-Docker
        if (-not (Test-Path 'web/node_modules')) {
            Write-Host '> first run: installing the console dependencies' -ForegroundColor DarkCyan
            Push-Location web
            try { & npm install; if ($LASTEXITCODE -ne 0) { throw 'npm install failed' } }
            finally { Pop-Location }
        }

        # --wait blocks until every healthcheck passes, so the services below
        # never start against a Postgres that is still booting.
        Invoke-Checked 'docker' ($Compose + @('up', '-d', '--wait'))

        $root = $PSScriptRoot
        # No ";" in any Command: wt.exe splits tabs on ";" even inside quotes,
        # which is why the console tab gets its directory from Dir instead of
        # a "Set-Location web; ..." prefix.
        $services = @(
            @{ Title = 'Beacon gateway :8000';   Dir = $root;            Command = '.\tasks.ps1 gateway' },
            @{ Title = 'Beacon retrieval :8001'; Dir = $root;            Command = '.\tasks.ps1 retrieval' },
            @{ Title = 'Beacon worker (bills)';  Dir = $root;            Command = '.\tasks.ps1 worker' },
            @{ Title = 'Beacon console :3000';   Dir = "$root\web";      Command = 'npm run dev' }
        )

        if (Get-Command wt.exe -ErrorAction SilentlyContinue) {
            # wt parses ";" itself to separate tabs, so the whole command line
            # is handed over as one string rather than through PowerShell.
            $tabs = foreach ($s in $services) {
                "new-tab --title `"$($s.Title)`" --suppressApplicationTitle -d `"$($s.Dir)`" " +
                "powershell -NoExit -NoProfile -ExecutionPolicy Bypass -Command `"$($s.Command)`""
            }
            Start-Process 'wt.exe' -ArgumentList ('-w new ' + ($tabs -join ' ; '))
        } else {
            foreach ($s in $services) {
                $cmd = "Set-Location -LiteralPath '$($s.Dir)'; `$host.UI.RawUI.WindowTitle = '$($s.Title)'; $($s.Command)"
                Start-Process 'powershell' -ArgumentList @('-NoExit', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', $cmd)
            }
        }

        # Open the console once it answers, not before: Next.js needs a few
        # seconds to compile, and opening early shows a connection error.
        Write-Host '> waiting for the console on http://localhost:3000 ...' -ForegroundColor DarkCyan
        $ready = $false
        for ($i = 0; $i -lt 90; $i++) {
            try {
                Invoke-WebRequest -Uri 'http://localhost:3000' -UseBasicParsing -TimeoutSec 2 | Out-Null
                $ready = $true; break
            } catch { Start-Sleep -Seconds 2 }
        }
        if ($ready) {
            Start-Process 'http://localhost:3000'
            Write-Host 'Beacon is up: http://localhost:3000' -ForegroundColor Green
        } else {
            Write-Host 'The console did not answer within 3 minutes - check the "Beacon console" tab.' -ForegroundColor Yellow
        }
        Write-Host 'The worker spends API credits only when you click Match trials. Close its tab to make that impossible.' -ForegroundColor Yellow
    }
}
