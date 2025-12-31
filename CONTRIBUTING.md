## Contributing

Thank you for your interest in contributing to the GNN Prompt Injection Detection System!

### Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/gauravj226/GNN_Prompt_Injection_Detection.git`
3. Create a virtual environment: `python -m venv venv`
4. Activate it: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
5. Install dependencies: `pip install -r requirements.txt`

### Development Workflow

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes with clear commit messages
3. Test your changes: `python -m pytest tests/`
4. Ensure code style compliance: `flake8 gnn_injection_detector.py`
5. Push to your fork and submit a pull request

### Code Style

- Follow PEP 8 guidelines
- Use descriptive variable and function names
- Add docstrings to all classes and functions
- Include type hints where applicable
- Keep functions focused and modular

### Testing

- Write tests for new features
- Ensure all tests pass before submitting PR
- Test with both CPU and GPU if possible
- Check for memory leaks in long-running code

### Commit Messages

- Use clear, descriptive commit messages
- Start with a verb (Add, Fix, Improve, Refactor)
- Example: "Add graph visualization function for text analysis"

### Pull Request Process

1. Update documentation for any changes
2. Add tests for new functionality
3. Update README.md with usage examples if needed
4. Ensure CI/CD passes
5. Wait for code review before merging

### Reporting Issues

- Use GitHub issues for bug reports
- Provide clear description and reproduction steps
- Include system information (OS, Python version, GPU info)
- Share minimal reproducible example if applicable

### Questions?

Feel free to open a discussion or issue if you have questions about the codebase!
