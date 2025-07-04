# 🛠️ Development Scripts - FluxForge Studio

This directory contains development and testing scripts for FluxForge Studio.

## 📁 Contents

### 🧪 **Test Scripts**

#### `test_flux_redux_integration.py`
- **Purpose**: Validates FLUX Redux integration
- **Usage**: `python test_flux_redux_integration.py`
- **Function**: Tests imports, database functions, and UI visibility controls

#### `test_model_cache_update.py`
- **Purpose**: Validates model cache list completeness
- **Usage**: `python test_model_cache_update.py`
- **Function**: Ensures all models used in the application are tracked in cache management

## 🚀 Usage

These scripts are primarily for development and testing purposes. They help ensure:
- ✅ All integrations work correctly
- ✅ No missing dependencies
- ✅ Architecture consistency
- ✅ Model cache completeness

## 📝 Notes

- These scripts may require specific dependencies that aren't needed for production
- They're designed to be run from the project root directory
- Results help validate the integrity of the FluxForge Studio codebase

## 🔧 Development Workflow

1. Make changes to core modules
2. Run relevant test scripts
3. Verify all tests pass
4. Deploy changes to production

These scripts complement the main utility scripts in the root directory and provide additional validation for development workflows.