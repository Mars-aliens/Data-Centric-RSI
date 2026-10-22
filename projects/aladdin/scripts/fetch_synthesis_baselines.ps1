$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Target = Join-Path $Root "baselines/synthesis"
$Sources = @(
    @{
        Name = "self-instruct-upstream"
        Url = "https://github.com/yizhongw/self-instruct.git"
        Revision = "0b26ccaa415992100fa32df62d41b994cf928e23"
        SparsePatterns = @("/README.md", "/LICENSE", "/requirements.txt", "/scripts/", "/self_instruct/")
    },
    @{
        Name = "wizardlm-upstream"
        Url = "https://github.com/nlpxucan/WizardLM.git"
        Revision = "cd4b4cc022b6426aee62d51f7ae99a4a6ec9aaef"
        SparsePatterns = @("/README.md", "/Evol_Instruct/", "/WizardLM/CODE_LICENSE")
    }
)

New-Item -ItemType Directory -Force -Path $Target | Out-Null
foreach ($Source in $Sources) {
    $Path = Join-Path $Target $Source.Name
    if (Test-Path (Join-Path $Path ".git")) {
        $Current = (& git -C $Path rev-parse HEAD).Trim()
        if ($Current -ne $Source.Revision) {
            throw "$Path is already present at $Current; expected $($Source.Revision)."
        }
        continue
    }

    & git clone --filter=blob:none --no-checkout $Source.Url $Path
    if ($LASTEXITCODE -ne 0) { throw "git clone failed for $($Source.Name)" }
    & git -C $Path sparse-checkout init --no-cone
    if ($LASTEXITCODE -ne 0) { throw "sparse-checkout init failed for $($Source.Name)" }
    & git -C $Path sparse-checkout set --no-cone -- $Source.SparsePatterns
    if ($LASTEXITCODE -ne 0) { throw "sparse-checkout set failed for $($Source.Name)" }
    & git -C $Path checkout $Source.Revision
    if ($LASTEXITCODE -ne 0) { throw "checkout failed for $($Source.Name)" }
}

Write-Host "Pinned synthesis baseline sources are ready under $Target"
