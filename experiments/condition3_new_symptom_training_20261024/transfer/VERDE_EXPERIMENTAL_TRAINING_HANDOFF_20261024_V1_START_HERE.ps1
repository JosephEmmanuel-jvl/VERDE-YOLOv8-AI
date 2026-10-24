param(
    [string]$Python = 'python',
    [string]$TransferDirectory = $PSScriptRoot,
    [string]$DestinationParent = (Join-Path $env:USERPROFILE 'VERDE_TRANSFERS')
)
$ErrorActionPreference = 'Stop'
$packageId = 'VERDE_EXPERIMENTAL_TRAINING_HANDOFF_20261024_V1'
$expected = 'f13c14179ee732714641e0b4b8715aebac1a8a6247ebd96662819f49f8aa1a38'
$archive = Join-Path $TransferDirectory ($packageId + '.zip')
if (-not (Test-Path -LiteralPath $archive -PathType Leaf)) { throw "ZIP missing: $archive" }
$hashAlgorithm = [System.Security.Cryptography.SHA256]::Create()
$sourceStream = [System.IO.File]::OpenRead($archive)
try { $actual = [System.BitConverter]::ToString($hashAlgorithm.ComputeHash($sourceStream)).Replace('-', '').ToLowerInvariant() }
finally { $sourceStream.Dispose(); $hashAlgorithm.Dispose() }
if ($actual -ne $expected) { throw "ZIP SHA-256 MISMATCH. Expected $expected; got $actual. Nothing extracted." }
Write-Host 'ZIP SHA-256 PASS. No installation or training will run.'
$destination = Join-Path $DestinationParent $packageId
if (Test-Path -LiteralPath $destination) { throw "Destination already exists: $destination. Use a NEW DestinationParent or manually run the included verifier on the existing copy." }
if (-not (Test-Path -LiteralPath $DestinationParent -PathType Container)) { New-Item -ItemType Directory -Path $DestinationParent | Out-Null }
& $Python -B -c "import pathlib,sys,zipfile; z=zipfile.ZipFile(sys.argv[1]); prefix=sys.argv[3]+'/'; names=z.namelist(); assert all(n.startswith(prefix) and '..' not in pathlib.PurePosixPath(n).parts and chr(92) not in n for n in names), 'Unsafe ZIP member'; assert len(names)==len(set(names)), 'Duplicate ZIP members'; assert all(len(str(pathlib.Path(sys.argv[2])/n))<260 for n in names), 'Use a shorter DestinationParent for Windows paths'; z.extractall(sys.argv[2]); z.close()" $archive $DestinationParent $packageId
if ($LASTEXITCODE -ne 0) { throw 'Extraction failed. No training was started.' }
$verifier = Join-Path $destination 'scripts\verify_transfer.py'
& $Python -B $verifier --zip $archive --expected-zip-sha256 $expected
if ($LASTEXITCODE -ne 0) { throw 'VERDE verification FAILED. Stop and resolve the reported mismatch or dependency issue.' }
Write-Host "VERDE transfer verification PASS: $destination"
Write-Host 'STOP: no training executed. Future candidate is EXPERIMENTAL_CONDITION3_CANDIDATE, NOT DEPLOYMENT-READY.'
