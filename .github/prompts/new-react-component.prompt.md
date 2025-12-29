---
description: Aggiungi nuovo componente React
---

# Nuovo Componente React

Crea un componente React seguendo le convenzioni del progetto NICO.

## Struttura File

```
frontend/src/components/
├── ComponentName/
│   ├── index.tsx        # Export
│   ├── ComponentName.tsx # Component
│   └── ComponentName.css # Styles (se non Tailwind)
```

## Template Componente

```tsx
import { useState, useEffect } from 'react';

interface ComponentNameProps {
  title: string;
  data?: DataType[];
  onAction?: (id: string) => void;
}

export function ComponentName({ 
  title, 
  data = [], 
  onAction 
}: ComponentNameProps) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Effect logic
  }, []);

  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error}</div>;

  return (
    <div className="p-4 bg-white rounded-lg shadow">
      <h2 className="text-xl font-bold">{title}</h2>
      {/* Content */}
    </div>
  );
}
```

## Convenzioni

- **Naming**: PascalCase per componenti
- **Props**: Interface con suffisso `Props`
- **Hooks**: Custom hooks in `hooks/`
- **State**: Zustand per state globale
- **Styling**: TailwindCSS preferito

## Checklist

- [ ] TypeScript interfaces definite
- [ ] Loading state gestito
- [ ] Error state gestito
- [ ] Props con default values
- [ ] Export dal barrel file
