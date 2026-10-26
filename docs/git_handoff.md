# Git delivery and integration

The repository is locally committed and delivered as a portable bundle. It has not been pushed to GitHub: branch creation through the connected GitHub tool was rejected because approval is required and this session's approval policy is never. Local `gh auth status` also reported an invalid existing credential; local outbound GitHub access was blocked. No credential was read or printed.

## Inspect the standalone repository

```sh
git clone h-potential-companions.bundle h-potential-review
cd h-potential-review
git log --oneline
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python analysis/reproduce.py
python analysis/build_papers.py
```

## Add to the existing repository after restoring GitHub write access

Run these commands from a folder containing the bundle, with authenticated Git access. They create an additive directory and a research branch; the existing notebook is retained.

```sh
git clone https://github.com/Nube1/Biomedical-Potential-Calculator-.git biomedical-research
git clone h-potential-companions.bundle h-potential-import
git -C h-potential-import archive --format=tar --output=../companions-import.tar HEAD
cd biomedical-research
git switch -c research/h-potential-thesis-companions
mkdir thesis_companions
tar -xf ../companions-import.tar -C thesis_companions
git add thesis_companions
git commit -m "Add H-potential companion papers and verified research code"
git push -u origin research/h-potential-thesis-companions
```

The included standalone workflow expects the project at repository root. When integrating in a subdirectory, move the workflow to the existing repository's .github/workflows and set each run step's working-directory to thesis_companions; update artifact paths accordingly. Do not imply that local tests prove remote CI has run.

Do not force-push or replace existing main history with the standalone bundle. A new dedicated repository is also possible if the owner chooses it. No repository was created or renamed during this session.
