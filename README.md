# Litematica Tools

Python tools for parsing and manipulating Minecraft Litematica schematic files.

## Features

- **Parse and display** Litematica schematic information with detailed block counts
- **Rotate schematics** by arbitrary angles around a specified pivot point with multiple rounding methods
- **Visualize schematics** with 2D overhead plots for visual verification
- **Swap block types** to quickly change materials in a schematic
- Automatically calculates and displays schematic center point
- Automatically updates directional block properties (stairs, signs, etc.)
- Merges block counts across orientations for accurate material lists
- Preserves all block data and metadata

## Requirements

- Python 3.7+
- Standard library only for core features
- `matplotlib` for plotting (optional): `sudo apt install python3-matplotlib`

## Commands Overview

```bash
python3 parse_litematic.py {info,rotate,plot,swap} [options]
```

- **`info`** - Display detailed schematic information
- **`rotate`** - Rotate a schematic around a pivot point
- **`plot`** - Create 2D overhead visualization
- **`swap`** - Replace one block type with another

## Usage

### Display Schematic Information

```bash
# Show detailed information about a schematic
python3 parse_litematic.py info your_schematic.litematic

# Alternative syntax
python3 parse_litematic.py info -i your_schematic.litematic
```

This will display:
- Metadata (name, author, description, timestamps)
- Dimensions and total volume
- **Center point coordinates** (useful for rotation)
- **Detailed block usage** with counts and percentages
- Tile entities and entities

### Rotate a Schematic

```bash
# Rotate 90 degrees clockwise around origin (0, 0)
python3 parse_litematic.py rotate your_schematic.litematic 90

# Rotate 45 degrees around a custom pivot point with smooth method
python3 parse_litematic.py rotate your_schematic.litematic 45 -x 100 -z 200 -m nearest_half

# Rotate -90 degrees (counter-clockwise)
python3 parse_litematic.py rotate your_schematic.litematic -90

# Alternative flag syntax with custom output
python3 parse_litematic.py rotate -i input.litematic -r 180 -o output.litematic
```

### Plot a Schematic

```bash
# Display overhead visualization
python3 parse_litematic.py plot your_schematic.litematic

# Save to file without displaying
python3 parse_litematic.py plot your_schematic.litematic --save --no-show

# Show specific Y level
python3 parse_litematic.py plot your_schematic.litematic -y 4 -o layer4.png --no-show
```

### Swap Block Types

```bash
# Swap stone for dirt
python3 parse_litematic.py swap structure.litematic stone dirt

# Swap wool colors
python3 parse_litematic.py swap structure.litematic orange_wool yellow_wool

# Alternative flag syntax
python3 parse_litematic.py swap -i structure.litematic -f stone -t dirt -o modified.litematic
```

## Commands

### `info` - Display Schematic Information

```bash
python3 parse_litematic.py info [options] <schematic_file>
```

**Options:**
- Positional: `<schematic_file>` - Input litematic file
- `-i, --input-file` - Input litematic file (alternative syntax)
- `-h, --help` - Show help for info command

**Displays:**
- Metadata (name, author, timestamps)
- Dimensions and center point
- Detailed block counts with percentages
- Block types merged across orientations

### `rotate` - Rotate a Schematic

```bash
python3 parse_litematic.py rotate [options] <schematic_file> <angle>
```

**Options:**
- Positional: `<schematic_file>` - Input litematic file
- Positional: `<angle>` - Rotation angle in degrees
- `-i, --input-file` - Input litematic file (alternative syntax)
- `-r, --rotate` - Rotation angle (alternative syntax)
- `-x` - X coordinate of pivot point (default: 0) - East/West axis
- `-z` - Z coordinate of pivot point (default: 0) - North/South axis
- `-m, --method` - Rotation method: `round` (default), `floor`, `ceil`, `nearest_half`
- `-o, --output` - Output file path (default: `<input>_rotated.litematic`)
- `-h, --help` - Show help for rotate command

**Rotation Methods:**
- `round` - Standard rounding (may cause dithering on angled rotations)
- `floor` - Always round down
- `ceil` - Always round up
- `nearest_half` - **Recommended for smooth results** on non-90° rotations

### `plot` - Visualize a Schematic

```bash
python3 parse_litematic.py plot [options] <schematic_file>
```

**Options:**
- Positional: `<schematic_file>` - Input litematic file
- `-i, --input-file` - Input litematic file (alternative syntax)
- `-o, --output` - Output image file path
- `--save` - Save plot as `<input>_plot.png`
- `-y, --y-level` - Only show blocks at specific Y level
- `--no-show` - Don't display window (only save)
- `-h, --help` - Show help for plot command

**Features:**
- Color-coded blocks by type
- Legend with top 15 most common blocks
- Useful for verifying rotations visually
- Requires matplotlib

### `swap` - Replace Block Types

```bash
python3 parse_litematic.py swap [options] <schematic_file> <from_block> <to_block>
```

**Options:**
- Positional: `<schematic_file>` - Input litematic file
- Positional: `<from_block>` - Block type to replace
- Positional: `<to_block>` - Block type to replace with
- `-i, --input-file` - Input litematic file (alternative syntax)
- `-f, --from` - Block to replace (alternative syntax)
- `-t, --to` - Replacement block (alternative syntax)
- `-o, --output` - Output file path (default: `<input>_swapped.litematic`)
- `-h, --help` - Show help for swap command

**Notes:**
- Block names can include or omit `minecraft:` namespace
- Block properties are preserved when compatible

## Coordinate System

In Minecraft:
- **X axis** = East (+) / West (-)
- **Y axis** = Up (+) / Down (-)
- **Z axis** = South (+) / North (-)

Rotation is performed in the horizontal plane (XZ), around the Y axis.

## How Rotation Works

1. **Block Positions**: Each block's X and Z coordinates are rotated around the specified pivot point using standard 2D rotation matrix:
   ```
   new_x = x × cos(θ) - z × sin(θ)
   new_z = x × sin(θ) + z × cos(θ)
   ```
   Where θ is the rotation angle. This is a mathematically correct geometric rotation.

2. **Directional Properties**: Block states with directional properties (like `facing`, `axis`, `rotation`) are automatically updated to match the new orientation
3. **Bounding Box**: The schematic's dimensions are recalculated to fit all rotated blocks
4. **Block Data**: All block data is preserved and re-encoded into the Litematica format

**Example:** A 10×10 square from (-5,-5) to (5,5) rotated 45° around (0,0) produces a diamond shape with corners at (0,-7), (7,0), (0,7), and (-7,0).

## Supported Block Properties

The tool automatically handles rotation for these block properties:
- `facing` (north/south/east/west) - for blocks like stairs, hoppers, observers
- `axis` (x/y/z) - for blocks like logs, pillars
- `rotation` (0-15) - for blocks like signs, banners, player heads

## Examples

### Example 1: View schematic details

```bash
python3 parse_litematic.py info my_house.litematic
```

**Output includes:**
- Schematic dimensions and total blocks
- **Center point** for accurate rotation
- Detailed block counts merged across orientations
- Material percentages

### Example 2: Rotate with smooth method

```bash
# Standard rotation (may have jagged edges)
python3 parse_litematic.py rotate structure.litematic 45 -x 365 -z 354

# Smooth rotation (recommended for angled rotations)
python3 parse_litematic.py rotate structure.litematic 45 -x 365 -z 354 -m nearest_half
```

### Example 3: Visual comparison of rotations

```bash
# Rotate and plot to compare methods
python3 parse_litematic.py rotate building.litematic 45 -m round -o round.litematic
python3 parse_litematic.py rotate building.litematic 45 -m nearest_half -o smooth.litematic

# Create plots to compare
python3 parse_litematic.py plot round.litematic -y 4 -o round_plot.png --no-show
python3 parse_litematic.py plot smooth.litematic -y 4 -o smooth_plot.png --no-show
```

### Example 4: Swap block materials

```bash
# Change all stone to smooth stone
python3 parse_litematic.py swap castle.litematic stone smooth_stone

# Swap wool colors for different team
python3 parse_litematic.py swap banner.litematic red_wool blue_wool
```

### Example 5: Rotate around center of hexagonal structure

If your structure's center point is at (365, 3, 354) (shown in `info` output):

```bash
# Rotate 60° for hexagonal structure (360° / 6 = 60°)
python3 parse_litematic.py rotate structure.litematic 60 -x 365 -z 354 -m nearest_half
```

### Example 6: Create multiple rotated versions

```bash
python3 parse_litematic.py rotate tower.litematic 90 -o tower_90.litematic
python3 parse_litematic.py rotate tower.litematic 180 -o tower_180.litematic
python3 parse_litematic.py rotate tower.litematic 270 -o tower_270.litematic
```

## Block Counting Features

The `info` command provides detailed block analysis:

- **Merged orientations**: Blocks with different orientations (axis, facing, rotation) are counted together
- **Preserved properties**: Important properties like `type`, `waterlogged` are retained
- **Sorted by frequency**: Most common blocks shown first
- **Percentage calculations**: Shows what portion of the build each block represents
- **Air block tracking**: Separate count for air blocks

This gives you an accurate **material list** for building the structure!

## Technical Details

### File Format

Litematica files (`.litematic`) are gzip-compressed NBT (Named Binary Tag) files containing:
- Metadata (name, author, timestamps, dimensions)
- Regions with block data stored in a palette + compressed block state array
- Optional tile entities and entities

### Rotation Algorithm

1. Parse the compressed NBT structure
2. Decode the block palette and block state array
3. Extract all non-air blocks with their positions
4. Apply 2D rotation transformation to coordinates
5. Update directional block properties
6. Rebuild palette and encode blocks back to the Litematica format
7. Update metadata with new dimensions and block counts
8. Write compressed NBT output

### Performance

- Parsing is fast even for large schematics (tested with 22,000+ blocks)
- Memory usage scales with the number of blocks (not the volume)
- Most operations complete in under a second

## Limitations

- Y-axis rotation only (horizontal plane) - vertical rotations not supported
- Some complex block states may not rotate correctly (please report issues)
- Very large schematics (millions of blocks) may require significant memory

## License

Free to use and modify. No warranty provided.

## Contributing

Bug reports and improvements welcome!
