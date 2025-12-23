/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      // =====================================================
      // 🎨 SWISS DESIGN SYSTEM - Blue Emerald Oceanography
      // =====================================================
      
      colors: {
        // Primary: Blue (Ocean Deep)
        blue: {
          50: '#eff6ff',
          100: '#dbeafe',
          200: '#bfdbfe',
          300: '#93c5fd',
          400: '#60a5fa',
          500: '#2563eb',  // Primary
          600: '#1d4ed8',
          700: '#1e40af',
          800: '#1e3a8a',
          900: '#172554',
          950: '#0f172a',
        },
        
        // Secondary: Emerald (Ocean Life)
        emerald: {
          50: '#ecfdf5',
          100: '#d1fae5',
          200: '#a7f3d0',
          300: '#6ee7b7',
          400: '#34d399',
          500: '#10b981',
          600: '#059669',  // Secondary
          700: '#047857',
          800: '#065f46',
          900: '#064e3b',
          950: '#022c22',
        },
        
        // Neutral: Slate (Clean Swiss)
        slate: {
          50: '#f8fafc',
          100: '#f1f5f9',
          200: '#e2e8f0',
          300: '#cbd5e1',
          400: '#94a3b8',
          500: '#64748b',
          600: '#475569',
          700: '#334155',
          800: '#1e293b',
          900: '#0f172a',
          950: '#020617',
        },
        
        // Accent: Cyan (Ocean Surface)
        cyan: {
          400: '#22d3ee',
          500: '#06b6d4',
          600: '#0891b2',
        },
        
        // Semantic Colors
        ocean: {
          deep: '#1e3a8a',
          mid: '#2563eb',
          surface: '#06b6d4',
          foam: '#ecfdf5',
        },
        
        // Graph colors for causal links
        causal: {
          strong: '#059669',    // Strong positive
          moderate: '#2563eb',  // Moderate
          weak: '#94a3b8',      // Weak
          negative: '#dc2626',  // Negative correlation
        },
      },
      
      // PHI-based spacing (Golden Ratio: 1.618)
      // Base: 8px, multiplied by PHI ratios
      spacing: {
        'phi-xs': '5px',     // 8 / 1.618
        'phi-sm': '8px',     // Base
        'phi-md': '13px',    // 8 * 1.618
        'phi-lg': '21px',    // 13 * 1.618
        'phi-xl': '34px',    // 21 * 1.618
        'phi-2xl': '55px',   // 34 * 1.618
        'phi-3xl': '89px',   // 55 * 1.618
        'phi-4xl': '144px',  // 89 * 1.618
      },
      
      // Swiss Typography
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
        display: ['Inter', 'system-ui', 'sans-serif'],
      },
      
      // Type scale based on PHI
      fontSize: {
        'phi-xs': ['0.618rem', { lineHeight: '1.5' }],    // ~10px
        'phi-sm': ['0.786rem', { lineHeight: '1.5' }],    // ~12.5px
        'phi-base': ['1rem', { lineHeight: '1.618' }],    // 16px
        'phi-lg': ['1.272rem', { lineHeight: '1.5' }],    // ~20px
        'phi-xl': ['1.618rem', { lineHeight: '1.4' }],    // ~26px
        'phi-2xl': ['2.058rem', { lineHeight: '1.3' }],   // ~33px
        'phi-3xl': ['2.618rem', { lineHeight: '1.2' }],   // ~42px
        'phi-4xl': ['3.33rem', { lineHeight: '1.1' }],    // ~53px
        'phi-5xl': ['4.236rem', { lineHeight: '1.05' }],  // ~68px
      },
      
      // Grid based on Swiss design principles
      gridTemplateColumns: {
        'swiss-12': 'repeat(12, 1fr)',
        'swiss-sidebar': '280px 1fr',
        'swiss-main': '1fr 320px',
      },
      
      // Border radius - minimal for Swiss aesthetic
      borderRadius: {
        'swiss-sm': '2px',
        'swiss': '4px',
        'swiss-lg': '6px',
      },
      
      // Box shadow - subtle and functional
      boxShadow: {
        'swiss-sm': '0 1px 2px 0 rgb(0 0 0 / 0.05)',
        'swiss': '0 2px 4px -1px rgb(0 0 0 / 0.06), 0 1px 2px -1px rgb(0 0 0 / 0.04)',
        'swiss-lg': '0 4px 6px -1px rgb(0 0 0 / 0.07), 0 2px 4px -2px rgb(0 0 0 / 0.04)',
        'swiss-glow-blue': '0 0 20px -5px rgba(37, 99, 235, 0.3)',
        'swiss-glow-emerald': '0 0 20px -5px rgba(5, 150, 105, 0.3)',
      },
      
      // Animation
      animation: {
        'fade-in': 'fadeIn 0.2s ease-out',
        'slide-up': 'slideUp 0.3s ease-out',
        'slide-down': 'slideDown 0.3s ease-out',
        'pulse-subtle': 'pulseSubtle 2s ease-in-out infinite',
        'graph-appear': 'graphAppear 0.5s ease-out',
      },
      
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        slideDown: {
          '0%': { opacity: '0', transform: 'translateY(-10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        pulseSubtle: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.8' },
        },
        graphAppear: {
          '0%': { opacity: '0', transform: 'scale(0.95)' },
          '100%': { opacity: '1', transform: 'scale(1)' },
        },
      },
      
      // Z-index scale
      zIndex: {
        'sidebar': '40',
        'header': '50',
        'modal': '100',
        'tooltip': '110',
        'toast': '120',
      },
    },
  },
  plugins: [],
}
