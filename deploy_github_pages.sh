#!/bin/bash
# ============================================================
# NABAT-AI — GitHub Pages Deploy Script
# Run this once from your Mac inside the handwritten-poems folder
# ============================================================
set -e

REPO_NAME="nabat-ai"
GITHUB_USER="${1:-asljneibi}"   # pass your GitHub username as arg, or edit here

echo "🚀 Deploying NABAT-AI to GitHub Pages..."
echo "   GitHub user : $GITHUB_USER"
echo "   Repo name   : $REPO_NAME"
echo ""

# 1. Create the remote repo (requires gh CLI: brew install gh && gh auth login)
if command -v gh &>/dev/null; then
    echo "📦 Creating GitHub repository..."
    gh repo create "$REPO_NAME" --public --description "NABAT-AI: Khaleeji Nabati Poetry Digitization & RAG System" 2>/dev/null || echo "   (repo may already exist, continuing)"
else
    echo "⚠️  gh CLI not found. Create the repo manually at https://github.com/new"
    echo "   Name it: $REPO_NAME   Visibility: Public"
    echo "   Then press Enter to continue..."
    read -r
fi

# 2. Add remote and push both branches
git remote remove origin 2>/dev/null || true
git remote add origin "https://github.com/$GITHUB_USER/$REPO_NAME.git"

echo ""
echo "📤 Pushing main branch..."
git push -u origin main

echo "📤 Pushing gh-pages branch..."
git push -u origin gh-pages

# 3. Enable GitHub Pages on gh-pages branch
if command -v gh &>/dev/null; then
    echo ""
    echo "⚙️  Enabling GitHub Pages..."
    gh api "repos/$GITHUB_USER/$REPO_NAME/pages" \
        --method POST \
        --field source='{"branch":"gh-pages","path":"/"}' 2>/dev/null || \
    echo "   Pages may already be configured — check Settings → Pages in your repo."
fi

echo ""
echo "✅ Done! Your app will be live in ~60 seconds at:"
echo "   https://$GITHUB_USER.github.io/$REPO_NAME/"
echo ""
echo "   (If you see a 404, go to GitHub → Settings → Pages"
echo "    and set Source to: gh-pages branch, / (root))"
