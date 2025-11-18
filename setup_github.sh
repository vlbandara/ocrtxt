#!/bin/bash

# Script to set up GitHub repository and push code
# Usage: ./setup_github.sh YOUR_GITHUB_USERNAME REPO_NAME

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: ./setup_github.sh YOUR_GITHUB_USERNAME REPO_NAME"
    echo "Example: ./setup_github.sh myusername receipt-scanner"
    exit 1
fi

GITHUB_USERNAME=$1
REPO_NAME=$2

echo "Setting up GitHub repository..."
echo "Repository: $GITHUB_USERNAME/$REPO_NAME"

# Set branch to main
git branch -M main

# Add remote origin
git remote add origin https://github.com/$GITHUB_USERNAME/$REPO_NAME.git 2>/dev/null || \
git remote set-url origin https://github.com/$GITHUB_USERNAME/$REPO_NAME.git

echo ""
echo "✅ Remote added!"
echo ""
echo "Next steps:"
echo "1. Create the repository on GitHub:"
echo "   Go to https://github.com/new"
echo "   Repository name: $REPO_NAME"
echo "   DO NOT initialize with README, .gitignore, or license"
echo "   Click 'Create repository'"
echo ""
echo "2. Then run:"
echo "   git push -u origin main"
echo ""
echo "Or if you've already created the repo, run the push command now:"
read -p "Push to GitHub now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    git push -u origin main
    echo ""
    echo "✅ Code pushed to GitHub!"
    echo ""
    echo "Your repository is now available at:"
    echo "https://github.com/$GITHUB_USERNAME/$REPO_NAME"
fi

