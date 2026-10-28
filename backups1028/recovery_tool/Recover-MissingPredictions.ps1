#requires -Version 5.1
<#
Read-only preview by default. Use -Apply to write validated missing DRatings
run projections to G:\baseball\docs\win\baseball\mlb\00_intake\predictions.
Every edited source file is backed up under G:\baseball\backups1028 first.
The attached audit and predictions log must be beside this .ps1 file.
This script does NOT run game_id_pred.py or replace model probabilities.
#>
[CmdletBinding()]
param([switch]$Apply)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$Repo = 'G:\baseball'
$BackupRoot = 'G:\baseball\backups1028'
$Mlb = Join-Path $Repo 'docs\win\baseball\mlb'
$PredDir = Join-Path $Mlb '00_intake\predictions'
$GamesDir = Join-Path $Mlb '00_intake\games'
$BookDir = Join-Path $Mlb '00_intake\sportsbook'
$AuditPath = Join-Path $PSScriptRoot 'baseball_missing_games_log_match_audit.csv'
$LogPath = Join-Path $PSScriptRoot 'predictions_log.csv'
$ExpectedAuditSha256 = 'DBDDD764B60DA765905B111927E19C3A17235BE0B8D7C8CA27E5B8175CDCF9D2'
$ExpectedLogSha256 = 'B7561DD916A5BD6EE7B64ED88812D95191FFB8E34E6B1A1014E4D11F69152705'
$Culture = [Globalization.CultureInfo]::InvariantCulture
$Utf8 = [System.Text.UTF8Encoding]::new($false)
$Header = 'game_id,sport,league,game_date,game_time,home_team,away_team,home_pitcher,away_pitcher,home_prob,away_prob,away_projected_runs,home_projected_runs,total_projected_runs'
$Columns = @($Header.Split(','))
$Abbrev = @{
    ARI='Arizona Diamondbacks'; ATH='Athletics'; ATL='Atlanta Braves'; BAL='Baltimore Orioles'
    BOS='Boston Red Sox'; CHC='Chicago Cubs'; CHW='Chicago White Sox'; CIN='Cincinnati Reds'
    CLE='Cleveland Guardians'; COL='Colorado Rockies'; DET='Detroit Tigers'; HOU='Houston Astros'
    KCR='Kansas City Royals'; LAA='Los Angeles Angels'; LAD='Los Angeles Dodgers'
    MIA='Miami Marlins'; MIL='Milwaukee Brewers'; MIN='Minnesota Twins'; NYM='New York Mets'
    NYY='New York Yankees'; PHI='Philadelphia Phillies'; PIT='Pittsburgh Pirates'
    SDP='San Diego Padres'; SEA='Seattle Mariners'; SFG='San Francisco Giants'
    STL='St Louis Cardinals'; TBR='Tampa Bay Rays'; TEX='Texas Rangers'
    TOR='Toronto Blue Jays'; WSN='Washington Nationals'
}

function Fail([string]$Message) { throw $Message }
function Norm([object]$Value) {
    return (([string]$Value).ToLowerInvariant() -replace '[^a-z0-9]', '')
}
function CsvField([object]$Value) {
    $s = [string]$Value
    if ($s -match '[,"\r\n]') { return '"' + $s.Replace('"','""') + '"' }
    return $s
}
function CsvLine([object[]]$Cells) {
    return (($Cells | ForEach-Object { CsvField $_ }) -join ',')
}
function SaveReport([string]$Path, [object[]]$Rows) {
    $content = @($Rows | ConvertTo-Csv -NoTypeInformation) -join "`r`n"
    [IO.File]::WriteAllText($Path, $content + "`r`n", $Utf8)
}
function ParseLocal([string]$Date, [string]$Time) {
    $dateText = $Date.Replace('_','-')
    $raw = ($dateText + ' ' + $Time.Trim())
    $parsed = [datetime]::MinValue
    $formats = [string[]]@(
        'yyyy-MM-dd HH:mm:ss', 'yyyy-MM-dd HH:mm',
        'yyyy-MM-dd h:mm tt', 'yyyy-MM-dd hh:mm tt',
        'yyyy-MM-dd h:mm:ss tt', 'yyyy-MM-dd hh:mm:ss tt'
    )
    if ([datetime]::TryParseExact($raw, $formats, $Culture,
            [Globalization.DateTimeStyles]::None, [ref]$parsed)) { return $parsed }
    return $null
}
function ParseRun([object]$Value) {
    $v = [decimal]::Zero
    if (-not [decimal]::TryParse([string]$Value,
            [Globalization.NumberStyles]::Float, $Culture, [ref]$v)) { return $null }
    if ($v -lt 0 -or $v -gt 30) { return $null }
    return $v
}
function MatchTeam([string]$Raw, [string]$Abbreviation) {
    if (-not $Abbrev.ContainsKey($Abbreviation)) { return $false }
    $actual = Norm $Raw
    $canonical = Norm $Abbrev[$Abbreviation]
    return ($actual -eq $canonical -or
            ($Abbreviation -eq 'ATH' -and $actual -eq 'oaklandathletics') -or
            ($Abbreviation -eq 'STL' -and $actual -eq 'stlouiscardinals'))
}
function MatchPair([object]$Row, [object]$AuditRow) {
    return ((MatchTeam ([string]$Row.home_team) ([string]$AuditRow.home_abbrev)) -and
            (MatchTeam ([string]$Row.away_team) ([string]$AuditRow.away_abbrev)))
}
function SameLog([object]$AuditRow, [object]$Source) {
    foreach ($pair in @(
        @('log_logged_at','logged_at'), @('log_date','date'),
        @('log_game_number','game_number'), @('log_away_abbrev','away_abbrev'),
        @('log_home_abbrev','home_abbrev'), @('log_dratings_away_proj','dratings_away_proj'),
        @('log_dratings_home_proj','dratings_home_proj'),
        @('log_dratings_total_proj','dratings_total_proj')
    )) {
        if ([string]$AuditRow.($pair[0]) -cne [string]$Source.($pair[1])) { return $false }
    }
    return $true
}
function CoveredByExisting([object]$Candidate, [object[]]$Existing, [object[]]$Games, [string]$Date) {
    # Return EXISTING, CLEAR, or AMBIGUOUS. Conservative for doubleheaders.
    foreach ($p in $Existing) {
        if ([string]$p.game_id -eq [string]$Candidate.audit_game_id) { return 'EXISTING' }
    }
    $matching = @($Games | Where-Object { MatchPair $_ $Candidate })
    if ($matching.Count -eq 0) { return 'AMBIGUOUS' }
    $sameTeam = @($Existing | Where-Object { MatchPair $_ $Candidate })
    if ($sameTeam.Count -eq 0) { return 'CLEAR' }
    if ($matching.Count -eq 1) { return 'EXISTING' }

    $target = @($matching | Where-Object { [string]$_.game_id -eq [string]$Candidate.audit_game_id })
    if ($target.Count -ne 1) { return 'AMBIGUOUS' }
    foreach ($p in $sameTeam) {
        $pTime = ParseLocal $Date ([string]$p.game_time)
        if ($null -eq $pTime) { return 'AMBIGUOUS' }
        $distances = @()
        foreach ($g in $matching) {
            $gTime = ParseLocal $Date ([string]$g.game_time)
            if ($null -eq $gTime) { return 'AMBIGUOUS' }
            $distances += [pscustomobject]@{
                Id=[string]$g.game_id
                Minutes=[math]::Abs(($pTime - $gTime).TotalMinutes)
            }
        }
        $rank = @($distances | Sort-Object Minutes)
        if ($rank[0].Minutes -gt 120) { return 'AMBIGUOUS' }
        if ($rank.Count -gt 1 -and ($rank[1].Minutes - $rank[0].Minutes) -lt 15) {
            return 'AMBIGUOUS'
        }
        if ($rank[0].Id -eq [string]$Candidate.audit_game_id) { return 'EXISTING' }
    }
    return 'CLEAR'
}
function ReportRow([object]$Candidate, [string]$Status, [string]$Reason) {
    return [pscustomobject]@{
        game_date=[string]$Candidate.audit_game_date
        game_id=[string]$Candidate.audit_game_id
        game_number=[string]$Candidate.game_number
        away=[string]$Candidate.away_abbrev
        home=[string]$Candidate.home_abbrev
        log_game_id=[string]$Candidate.log_game_id
        logged_at=[string]$Candidate.log_logged_at
        sportsbook_match=$(if ($BookResults.ContainsKey([string]$Candidate.audit_game_id)) { $BookResults[[string]$Candidate.audit_game_id] } else { 'NOT_CHECKED' })
        status=$Status
        reason=$Reason
    }
}

foreach ($p in @($Repo, $PredDir, $GamesDir, $AuditPath, $LogPath)) {
    if (-not (Test-Path -LiteralPath $p)) { Fail "Required path missing: $p" }
}
if ((Get-FileHash -LiteralPath $AuditPath -Algorithm SHA256).Hash -ne $ExpectedAuditSha256 -or
    (Get-FileHash -LiteralPath $LogPath -Algorithm SHA256).Hash -ne $ExpectedLogSha256) {
    Fail 'Audit/log SHA256 mismatch; expected the reviewed source attachments.'
}
$RunDir = Join-Path $BackupRoot ('prediction_recovery_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
New-Item -ItemType Directory -Path $RunDir -Force | Out-Null
$Audit = @(Import-Csv -LiteralPath $AuditPath -Encoding UTF8)
$SourceLog = @(Import-Csv -LiteralPath $LogPath -Encoding UTF8)
if ($Audit.Count -ne 424 -or $SourceLog.Count -ne 1034) {
    Fail "Source files do not match reviewed audit/log row counts (424/1034); nothing written."
}
$LogById = @{}
foreach ($line in $SourceLog) {
    $id = [string]$line.game_id
    if ([string]::IsNullOrWhiteSpace($id) -or $LogById.ContainsKey($id)) {
        Fail "Missing/duplicate log game_id: $id"
    }
    $LogById[$id] = $line
}
$TimeZone = [TimeZoneInfo]::FindSystemTimeZoneById('Eastern Standard Time')
$Reports = [System.Collections.Generic.List[object]]::new()
$Plans = @{}
$seen = @{}
$GamesCache = @{}
$ExistingCache = @{}
$InitialHash = @{}
$BookCache = @{}
$BookResults = @{}

foreach ($r in $Audit) {
    $id = [string]$r.audit_game_id
    $date = [string]$r.audit_game_date
    if ($seen.ContainsKey($id)) { Fail "Duplicate audited game_id: $id" }
    $seen[$id] = $true
    if ($r.match_status -ne 'MATCHED') {
        $Reports.Add((ReportRow $r 'SKIP_NOT_MATCHED' ([string]$r.match_status)))
        continue
    }
    if ($r.dratings_runs_available -ne 'YES') {
        $Reports.Add((ReportRow $r 'SKIP_NO_DRATINGS_RUNS' 'Only other sources or incomplete DRatings projections'))
        continue
    }
    if (-not $LogById.ContainsKey([string]$r.log_game_id) -or
        -not (SameLog $r $LogById[[string]$r.log_game_id])) {
        $Reports.Add((ReportRow $r 'SKIP_LOG_MISMATCH' 'Audit/log values do not agree'))
        continue
    }
    if ($r.log_date -ne $date -or
        $r.log_game_number -ne $r.game_number -or
        $r.log_away_abbrev -ne $r.away_abbrev -or
        $r.log_home_abbrev -ne $r.home_abbrev) {
        $Reports.Add((ReportRow $r 'SKIP_IDENTITY_MISMATCH' 'Log date/teams/game number differ'))
        continue
    }
    if ($date -notmatch '^2026-\d{2}-\d{2}$' -or $id -notmatch '^\d+$' -or
        $r.game_number -notmatch '^\d+$' -or
        -not $Abbrev.ContainsKey([string]$r.home_abbrev) -or
        -not $Abbrev.ContainsKey([string]$r.away_abbrev)) {
        $Reports.Add((ReportRow $r 'SKIP_UNTRUSTED_IDENTITY' 'Unexpected identifier/date/team'))
        continue
    }
    $away = ParseRun $r.log_dratings_away_proj
    $homeRuns = ParseRun $r.log_dratings_home_proj
    $total = ParseRun $r.log_dratings_total_proj
    if ($null -eq $away -or $null -eq $homeRuns -or $null -eq $total -or
        [math]::Abs([double]($away + $homeRuns - $total)) -gt 0.03) {
        $Reports.Add((ReportRow $r 'SKIP_INVALID_RUNS' 'Missing/invalid projection or inconsistent total'))
        continue
    }
    $dateUnderscore = $date.Replace('-','_')
    $target = Join-Path $PredDir ($dateUnderscore + '_MLB.csv')
    $expectedRelative = 'docs/win/baseball/mlb/00_intake/predictions/' + $dateUnderscore + '_MLB.csv'
    if ([string]$r.target_prediction_path -ne $expectedRelative) {
        $Reports.Add((ReportRow $r 'SKIP_TARGET_MISMATCH' 'Unexpected audit target path'))
        continue
    }
    if (-not $GamesCache.ContainsKey($date)) {
        $gamePath = Join-Path $GamesDir ($dateUnderscore + '_games.csv')
        $GamesCache[$date] = if (Test-Path -LiteralPath $gamePath) {
            @(Import-Csv -LiteralPath $gamePath -Encoding UTF8)
        } else { @() }
    }
    $games = @($GamesCache[$date])
    $gameMatches = @($games | Where-Object { [string]$_.game_id -eq $id })
    if ($gameMatches.Count -ne 1) {
        $Reports.Add((ReportRow $r 'SKIP_GAME_ID_NOT_UNIQUE' 'Official games row missing or duplicate'))
        continue
    }
    $g = $gameMatches[0]
    if (-not (MatchPair $g $r) -or ([string]$g.game_date).Replace('_','-') -ne $date -or
        [string]$g.gameNumber -ne [string]$r.game_number -or
        [string]::IsNullOrWhiteSpace([string]$g.gamePk)) {
        $Reports.Add((ReportRow $r 'SKIP_GAME_IDENTITY_CONFLICT' 'Official games teams/number/gamePk mismatch'))
        continue
    }
    if (-not $BookCache.ContainsKey($date)) {
        $bookPath = Join-Path $BookDir ($dateUnderscore + '_MLB.csv')
        $BookCache[$date] = if (Test-Path -LiteralPath $bookPath) {
            @(Import-Csv -LiteralPath $bookPath -Encoding UTF8)
        } else { @() }
    }
    $books = @($BookCache[$date] | Where-Object { [string]$_.game_id -eq $id })
    $BookResults[$id] = if ($books.Count -eq 1 -and (MatchPair $books[0] $r)) {
        'MATCHED'
    } elseif ($books.Count -eq 0) { 'MISSING' } else { 'CONFLICT' }
    $startTime = ParseLocal $date ([string]$g.game_time)
    if ($null -eq $startTime) {
        $Reports.Add((ReportRow $r 'SKIP_UNKNOWN_FIRST_PITCH' 'Official games time missing/unparseable'))
        continue
    }
    $loggedAt = [DateTimeOffset]::MinValue
    if (-not [DateTimeOffset]::TryParse([string]$r.log_logged_at, $Culture,
            [Globalization.DateTimeStyles]::None, [ref]$loggedAt)) {
        $Reports.Add((ReportRow $r 'SKIP_BAD_LOG_TIMESTAMP' 'Cannot verify pregame capture'))
        continue
    }
    $firstPitch = [DateTimeOffset]::new($startTime, $TimeZone.GetUtcOffset($startTime))
    if ($loggedAt -ge $firstPitch) {
        $Reports.Add((ReportRow $r 'SKIP_POST_FIRST_PITCH' 'Logged at/after official scheduled start'))
        continue
    }
    if (-not $ExistingCache.ContainsKey($date)) {
        if (Test-Path -LiteralPath $target) {
            $InitialHash[$date] = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash
            $firstLine = [IO.File]::ReadLines($target) | Select-Object -First 1
            if ($firstLine.TrimStart([char]0xFEFF) -cne $Header) {
                Fail "Refusing to overwrite unexpected CSV schema: $target"
            }
            $ExistingCache[$date] = [System.Collections.ArrayList]::new()
            foreach ($p in @(Import-Csv -LiteralPath $target -Encoding UTF8)) {
                [void]$ExistingCache[$date].Add($p)
            }
        } else {
            $InitialHash[$date] = ''
            $ExistingCache[$date] = [System.Collections.ArrayList]::new()
        }
    }
    $existing = @($ExistingCache[$date].ToArray())
    $covered = CoveredByExisting $r $existing $games $date
    if ($covered -eq 'EXISTING') {
        $Reports.Add((ReportRow $r 'SKIP_ALREADY_PRESENT' 'Existing prediction protected'))
        continue
    }
    if ($covered -ne 'CLEAR') {
        $Reports.Add((ReportRow $r 'SKIP_AMBIGUOUS_MATCH' 'Could duplicate a doubleheader or unmatched source row'))
        continue
    }
    $gameTime = $startTime.ToString('hh:mm tt', $Culture)
    $homeAbbr = [string]$r.home_abbrev
    $awayAbbr = [string]$r.away_abbrev
    $values = @(
        $id, 'baseball', 'mlb', $dateUnderscore, $gameTime,
        $Abbrev[$homeAbbr], $Abbrev[$awayAbbr],
        '', '', '', '',
        $away.ToString($Culture), $homeRuns.ToString($Culture), $total.ToString($Culture)
    )
    $line = CsvLine $values
    if (-not $Plans.ContainsKey($date)) { $Plans[$date] = [System.Collections.ArrayList]::new() }
    [void]$Plans[$date].Add([pscustomobject]@{ Target=$target; Line=$line; GameId=$id })
    $synthetic = [pscustomobject]@{
        game_id=$id; home_team=$Abbrev[$homeAbbr]
        away_team=$Abbrev[$awayAbbr]; game_time=$gameTime
    }
    [void]$ExistingCache[$date].Add($synthetic)
    $Reports.Add((ReportRow $r $(if ($Apply) {'READY_TO_ADD'} else {'WOULD_ADD'}) 'Verified pregame DRatings projections; win probabilities blank'))
}

$ReportPath = Join-Path $RunDir 'recovery_report.csv'
SaveReport $ReportPath @($Reports.ToArray())
$PlanCount = @($Reports | Where-Object { $_.status -in @('READY_TO_ADD', 'WOULD_ADD') }).Count
Write-Host "Candidate additions: $PlanCount; date files: $($Plans.Count)"
$blockedByBook = @($Reports | Where-Object { $_.status -in @('READY_TO_ADD', 'WOULD_ADD') -and $_.sportsbook_match -ne 'MATCHED' }).Count
Write-Host "Candidates not matched to sportsbook: $blockedByBook (downstream game_id_pred.py may reject them)"
Write-Host "Review report: $ReportPath"
if (-not $Apply) {
    Write-Host 'PREVIEW ONLY. No prediction CSVs modified. Review report, then rerun with -Apply.'
    exit 0
}

$BackupDir = Join-Path $RunDir 'original_files'
New-Item -ItemType Directory -Path $BackupDir -Force | Out-Null
# Back up ALL affected files BEFORE changing any of them.
foreach ($date in @($Plans.Keys | Sort-Object)) {
    $path = $Plans[$date][0].Target
    if (Test-Path -LiteralPath $path) {
        Copy-Item -LiteralPath $path -Destination (Join-Path $BackupDir ([IO.Path]::GetFileName($path))) -ErrorAction Stop
        $originalHash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
        $backupHash = (Get-FileHash -LiteralPath (Join-Path $BackupDir ([IO.Path]::GetFileName($path))) -Algorithm SHA256).Hash
        if ($originalHash -ne $backupHash) { Fail "Backup checksum mismatch: $path" }
    }
}
$changed = 0
foreach ($date in @($Plans.Keys | Sort-Object)) {
    $items = @($Plans[$date].ToArray())
    $path = [string]$items[0].Target
    $temp = $path + '.missing_predictions_recovery.tmp'
    if (Test-Path -LiteralPath $temp) { Fail "Refusing to replace existing temp file: $temp" }
    $exists = Test-Path -LiteralPath $path
    $expectedHash = [string]$InitialHash[$date]
    if (($exists -and ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $expectedHash)) -or
        ((-not $exists) -and $expectedHash -ne '')) {
        Fail "Source file changed after audit; refusing edit: $path"
    }
    try {
        $bytes = if ($exists) { [IO.File]::ReadAllBytes($path) } else { [byte[]]@() }
        $prefix = if (-not $exists) { $Header + "`r`n" }
                  elseif ($bytes.Length -gt 0 -and $bytes[$bytes.Length - 1] -ne 10) { "`r`n" }
                  else { '' }
        $tail = $prefix + (($items | ForEach-Object { $_.Line }) -join "`r`n") + "`r`n"
        $tailBytes = $Utf8.GetBytes($tail)
        $stream = [IO.File]::Open($temp, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write)
        try {
            $stream.Write($bytes, 0, $bytes.Length)
            $stream.Write($tailBytes, 0, $tailBytes.Length)
            $stream.Flush($true)
        } finally { $stream.Dispose() }
        if ($exists) {
            [IO.File]::Replace($temp, $path, $null)
        } else {
            [IO.File]::Move($temp, $path)
        }
        $changed++
    } finally {
        if (Test-Path -LiteralPath $temp) { Remove-Item -LiteralPath $temp -Force }
    }
}
foreach ($item in $Reports) { if ($item.status -eq 'READY_TO_ADD') { $item.status = 'ADDED' } }
SaveReport $ReportPath @($Reports.ToArray())
Write-Host "SUCCESS: appended $PlanCount prediction rows across $changed files."
Write-Host "Backups: $BackupDir"
Write-Host 'The downstream game_id_pred.py stage is NOT run; sportsbook-only eligible rows require a separate check.'
