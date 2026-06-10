# Contributing to Fire & Smoke Detection App

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to this project.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Code Style](#code-style)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)

---

## Code of Conduct

Please be respectful and constructive in all interactions. We are committed to providing a welcoming and inclusive experience for everyone.

---

## Getting Started

1. **Fork** the repository
2. **Clone** your fork:
   ```bash
   git clone https://github.com/shreyaa4567/fire-smoke-detection-app.git
   cd fire-smoke-detection-app
   ```
3. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/Mac
   .venv\Scripts\activate      # Windows
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
5. **Run the app** to verify setup:
   ```bash
   streamlit run appnew.py
   ```

---

## How to Contribute

### Types of Contributions Welcome

| Type | Description |
|------|-------------|
| 🐛 Bug Fixes | Fix existing issues or broken functionality |
| ✨ Features | Add new detection modes, UI improvements, or integrations |
| 📖 Documentation | Improve README, add tutorials, or fix typos |
| 🎨 UI/UX | Enhance the dashboard design or add new visualizations |
| ⚡ Performance | Optimize inference speed or memory usage |
| 🧪 Testing | Add test cases or testing infrastructure |

### Feature Suggestions

If you have ideas for new features, please open an issue first to discuss the approach before investing time in implementation.

---

## Code Style

### Python

- Follow **PEP 8** conventions
- Use **type hints** for function parameters and return values
- Add **docstrings** for all public functions (Google style)
- Keep functions focused — one function, one responsibility
- Use **constants** for configuration values (thresholds, paths, etc.)

### Streamlit

- Use `unsafe_allow_html=True` sparingly — only for custom CSS components
- Always provide unique `key` parameters for interactive widgets
- Use `st.session_state` for state management — never global variables
- Cache expensive operations with `@st.cache_resource` or `@st.cache_data`

### CSS

- Follow the existing dark industrial design system
- Use CSS custom properties for colors and spacing where possible
- Keep animations subtle — no flashy or distracting effects
- Ensure responsive design for different screen sizes

### Naming Conventions

```python
# Variables and functions: snake_case
fire_count = 0
def render_detection_results():

# Constants: UPPER_SNAKE_CASE
CONFIDENCE_THRESHOLD = 0.15
IOU_THRESHOLD = 0.45

# CSS classes: kebab-case
# .stat-card, .det-item, .risk-panel
```

---

## Pull Request Process

1. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** with clear, atomic commits:
   ```bash
   git commit -m "feat: add webcam continuous mode"
   git commit -m "fix: resolve video codec compatibility issue"
   ```

3. **Test thoroughly**:
   - Run the app and test all 5 modes
   - Verify the dark theme is preserved
   - Check for console errors in the browser
   - Test with different image/video formats

4. **Push** your branch and open a PR:
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Describe your changes** in the PR description:
   - What problem does this solve?
   - What changes were made?
   - How was it tested?
   - Screenshots (if UI changes)

### Commit Message Convention

```
type: short description

Types:
  feat     — New feature
  fix      — Bug fix
  docs     — Documentation only
  style    — Code style (formatting, no logic change)
  refactor — Code restructuring (no feature/fix)
  perf     — Performance improvement
  test     — Adding or updating tests
  chore    — Build process, dependencies, tooling
```

---

## Reporting Issues

When reporting a bug, please include:

1. **Environment**: OS, Python version, browser
2. **Steps to reproduce**: Exact sequence of actions
3. **Expected behavior**: What you expected to happen
4. **Actual behavior**: What actually happened
5. **Screenshots**: If applicable (especially for UI issues)
6. **Error logs**: Any error messages from the terminal

### Issue Template

```markdown
**Bug Description**
A clear description of the bug.

**Steps to Reproduce**
1. Open the app
2. Navigate to '...'
3. Click '...'
4. See error

**Expected Behavior**
What should happen.

**Environment**
- OS: Windows 11
- Python: 3.11
- Browser: Chrome 120
- Streamlit: 1.30.0

**Screenshots**
If applicable.
```

---

## Questions?

If you have questions about contributing, feel free to open a discussion or issue. We're happy to help!
