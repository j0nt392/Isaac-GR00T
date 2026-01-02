#!/usr/bin/env bash
set -e

CONFIG_FILE="./pyproject.toml"

# --- Helper for colored output ---
RED=$(tput setaf 1)
GREEN=$(tput setaf 2)
YELLOW=$(tput setaf 3)
RESET=$(tput sgr0)

# --- Check arguments ---
if [ $# -eq 0 ]; then
  echo "${RED}❌ Error:${RESET} No path provided."
  echo "Usage: $0 [--check] <path/to/file_or_dir>"
  exit 1
fi

CHECK_MODE=false
if [[ $1 == "--check" ]]; then
  CHECK_MODE=true
  shift
fi

TARGET=$1
if [ ! -e "$TARGET" ]; then
  echo "${RED}❌ Error:${RESET} Target '$TARGET' not found."
  exit 1
fi

echo ""
echo "🧩 Running formatter on: ${YELLOW}${TARGET}${RESET}"
echo ""

# --- Run Black ---
echo "🧠 Checking code style with black..."
if $CHECK_MODE; then
  if black --check --diff --config "$CONFIG_FILE" "$TARGET"; then
    BLACK_STATUS=0
  else
    BLACK_STATUS=$?
  fi
else
  black --config "$CONFIG_FILE" "$TARGET"
  BLACK_STATUS=$?
fi
echo ""

# --- Run isort ---
echo "📦 Checking imports with isort..."
if $CHECK_MODE; then
  if isort --settings-path "$CONFIG_FILE" --check-only --diff "$TARGET"; then
    ISORT_STATUS=0
  else
    ISORT_STATUS=$?
  fi
else
  isort --settings-path "$CONFIG_FILE" "$TARGET"
  ISORT_STATUS=$?
fi
echo ""

# --- Run Ruff ---
echo "🧹 Checking lint with ruff..."
if ! ruff check --config "$CONFIG_FILE" "$TARGET"; then
  echo "⚠️ Ruff found issues (not all are auto-fixable)."
  echo "💡 Tip: Run 'ruff check --fix' to auto-correct what can be fixed."
else
  echo "✅ Ruff passed."
fi

echo ""

# --- Summarize results ---
echo "📊 ${YELLOW}Summary:${RESET}"
if [ $BLACK_STATUS -eq 0 ]; then
  echo "✅ Black: No issues"
else
  echo "⚠️  Black found issues"
fi

if [ $ISORT_STATUS -eq 0 ]; then
  echo "✅ isort: No issues"
else
  echo "⚠️  isort found issues"
fi

if [ $RUFF_STATUS -eq 0 ]; then
  echo "✅ Ruff: No issues"
else
  echo "⚠️  Ruff found issues"
fi

echo ""
if $CHECK_MODE; then
  if [[ $BLACK_STATUS -eq 0 && $ISORT_STATUS -eq 0 && $RUFF_STATUS -eq 0 ]]; then
    echo "${GREEN}🎉 All checks passed!${RESET}"
  else
    echo "${RED}❌ Some checks failed. Please review the diffs above.${RESET}"
    exit 1
  fi
else
  echo "${GREEN}✅ Formatting and linting complete.${RESET}"
fi
