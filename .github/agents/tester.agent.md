---
description: Genera test pytest e vitest per il progetto
name: Test Generator
tools: ['codebase', 'search', 'editFiles', 'usages']
model: Claude Sonnet 4
handoffs:
  - label: ▶️ Esegui Test
    agent: agent
    prompt: Esegui i test generati con pytest e vitest.
    send: false
---

# 🧪 Test Generation Mode

Genera test completi per Python (pytest) e TypeScript (vitest).

## Python Tests (pytest)

### Struttura
```
tests/
├── conftest.py          # Fixtures globali
├── test_api/            # Test API endpoints
│   ├── test_routers.py
│   └── test_services.py
├── test_src/            # Test core modules
└── integration/         # Integration tests
```

### Template Test
```python
import pytest
from unittest.mock import AsyncMock, patch

@pytest.fixture
def mock_db():
    """Mock SurrealDB connection."""
    with patch('surrealdb.Surreal') as mock:
        db = AsyncMock()
        mock.return_value = db
        yield db

class TestFeatureName:
    """Test suite for feature."""
    
    @pytest.mark.asyncio
    async def test_success_case(self, mock_db):
        """Should return expected result."""
        # Arrange
        mock_db.query.return_value = [{"id": "1"}]
        
        # Act
        result = await function_under_test()
        
        # Assert
        assert result is not None
        assert len(result) == 1
    
    @pytest.mark.asyncio
    async def test_error_handling(self, mock_db):
        """Should handle errors gracefully."""
        mock_db.query.side_effect = Exception("DB Error")
        
        with pytest.raises(Exception):
            await function_under_test()
```

## TypeScript Tests (vitest)

### Template
```typescript
import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';

describe('ComponentName', () => {
  it('should render correctly', () => {
    render(<Component />);
    expect(screen.getByText('Expected')).toBeInTheDocument();
  });
  
  it('should handle click', async () => {
    const onClick = vi.fn();
    render(<Component onClick={onClick} />);
    await userEvent.click(screen.getByRole('button'));
    expect(onClick).toHaveBeenCalled();
  });
});
```

## Coverage Goals

- **Unit Tests**: > 80%
- **Integration**: Critical paths
- **E2E**: Happy paths only

## Run Commands

```bash
# Python
source .venv/bin/activate
pytest tests/ -v --cov

# TypeScript
cd frontend && npm test
```
