# Aivalon Frontend - Pixel Art Style

This project now features a pixelated art playful style for the Aivalon Transcript Viewer frontend. The design features:

## Design Elements

- **Pixel Art Aesthetic**: Using crisp edges and pixel-style rendering
- **Retro Gaming Colors**: Bold, saturated colors reminiscent of classic video games
- **8-bit Inspired Typography**: Using the "Press Start 2P" font
- **Pixel-Perfect UI Elements**: All interface elements follow a consistent pixel grid

## Key Visual Features

- Pixelated backgrounds with grid patterns
- Sharp corners and blocky elements instead of rounded ones
- Bright, contrasting colors with a limited palette
- Glow effects and animations that mimic retro gaming UIs
- Pixel-style buttons and interactive elements

## Technical Implementation

- Uses SCSS for enhanced CSS capabilities
- Custom variables for consistent pixel art styling
- CSS animations for subtle interactive feedback
- Pixel-perfect spacing using CSS variables

## Files Modified

- `src/pixel-styles.scss` - New pixel art styling
- `index.html` - Added Press Start 2P font import
- `main.tsx` - Updated to use the new styles
- `package.json` - Added sass dependency

## How to Run

```bash
cd frontend
npm install
npm run dev
```

Visit `http://localhost:5173` to see the pixel art style in action.

## Customization

To adjust the pixel art aesthetic, modify the CSS variables in the `:root` section of `src/pixel-styles.scss`:

- `--pixel-size`: Controls the base pixel size
- `--pixel-radius`: Controls corner radius (keep low for pixel effect)
- `--pixel-shadow`: Controls button and element shadows
- `--pixel-glow`: Controls glow effects for interactive elements