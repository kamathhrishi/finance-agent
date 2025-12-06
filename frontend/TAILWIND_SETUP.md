# Tailwind CSS Production Setup

## Current Status
The application currently uses Tailwind CSS via CDN, which is not recommended for production.

## Production Setup

To properly set up Tailwind CSS for production:

1. **Install Tailwind CSS:**
   ```bash
   npm install -D tailwindcss
   npx tailwindcss init
   ```

2. **Configure tailwind.config.js:**
   ```javascript
   module.exports = {
     content: ["./index.html", "./landing.html", "./**/*.{js,ts,jsx,tsx}"],
     theme: {
       extend: {
         colors: {
           'strata-blue': {
             50: '#f0f7ff', 100: '#e0eefe', 200: '#c2e0fd', 300: '#a3d1fc',
             400: '#6ab8f9', 500: '#329ef6', 600: '#0083f1', 700: '#0070d8',
             800: '#005cb6', 900: '#004f97', 950: '#00315e'
           },
           'strata-slate': {
             50: '#f8fafc', 100: '#f1f5f9', 200: '#e2e8f0', 300: '#cbd5e1',
             400: '#94a3b8', 500: '#64748b', 600: '#475569', 700: '#334155',
             800: '#1e293b', 900: '#0f172a', 950: '#020617'
           },
         },
         fontFamily: { 
           'sans': ['Inter', 'sans-serif']
         }
       }
     },
     plugins: []
   }
   ```

3. **Create input CSS file (src/input.css):**
   ```css
   @tailwind base;
   @tailwind components;
   @tailwind utilities;
   ```

4. **Build CSS:**
   ```bash
   npx tailwindcss -i ./src/input.css -o ./styles.css --watch
   ```

5. **Update HTML files:**
   - Remove: `<script src="https://cdn.tailwindcss.com"></script>`
   - Add: `<link rel="stylesheet" href="styles.css">`

## Benefits of Production Setup
- Smaller bundle size
- Better performance
- No CDN dependency
- Tree-shaking of unused styles
- Better caching
