#!/usr/bin/env python3
"""
Parse and display Minecraft Litematica schematic files
"""
import gzip
import struct
import argparse
import math
import time
from typing import Any, Dict, List, Tuple, Set
from pathlib import Path
from collections import defaultdict

class NBTParser:
    """Parse Minecraft NBT (Named Binary Tag) format"""

    TAG_END = 0
    TAG_BYTE = 1
    TAG_SHORT = 2
    TAG_INT = 3
    TAG_LONG = 4
    TAG_FLOAT = 5
    TAG_DOUBLE = 6
    TAG_BYTE_ARRAY = 7
    TAG_STRING = 8
    TAG_LIST = 9
    TAG_COMPOUND = 10
    TAG_INT_ARRAY = 11
    TAG_LONG_ARRAY = 12

    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def read_byte(self) -> int:
        value = struct.unpack('>b', self.data[self.pos:self.pos+1])[0]
        self.pos += 1
        return value

    def read_ubyte(self) -> int:
        value = struct.unpack('>B', self.data[self.pos:self.pos+1])[0]
        self.pos += 1
        return value

    def read_short(self) -> int:
        value = struct.unpack('>h', self.data[self.pos:self.pos+2])[0]
        self.pos += 2
        return value

    def read_int(self) -> int:
        value = struct.unpack('>i', self.data[self.pos:self.pos+4])[0]
        self.pos += 4
        return value

    def read_long(self) -> int:
        value = struct.unpack('>q', self.data[self.pos:self.pos+8])[0]
        self.pos += 8
        return value

    def read_float(self) -> float:
        value = struct.unpack('>f', self.data[self.pos:self.pos+4])[0]
        self.pos += 4
        return value

    def read_double(self) -> float:
        value = struct.unpack('>d', self.data[self.pos:self.pos+8])[0]
        self.pos += 8
        return value

    def read_string(self) -> str:
        length = self.read_short()
        value = self.data[self.pos:self.pos+length].decode('utf-8')
        self.pos += length
        return value

    def read_tag(self, tag_type: int) -> Any:
        if tag_type == self.TAG_END:
            return None
        elif tag_type == self.TAG_BYTE:
            return self.read_byte()
        elif tag_type == self.TAG_SHORT:
            return self.read_short()
        elif tag_type == self.TAG_INT:
            return self.read_int()
        elif tag_type == self.TAG_LONG:
            return self.read_long()
        elif tag_type == self.TAG_FLOAT:
            return self.read_float()
        elif tag_type == self.TAG_DOUBLE:
            return self.read_double()
        elif tag_type == self.TAG_BYTE_ARRAY:
            length = self.read_int()
            value = [self.read_byte() for _ in range(length)]
            return value
        elif tag_type == self.TAG_STRING:
            return self.read_string()
        elif tag_type == self.TAG_LIST:
            return self.read_list()
        elif tag_type == self.TAG_COMPOUND:
            return self.read_compound()
        elif tag_type == self.TAG_INT_ARRAY:
            length = self.read_int()
            value = [self.read_int() for _ in range(length)]
            return value
        elif tag_type == self.TAG_LONG_ARRAY:
            length = self.read_int()
            value = [self.read_long() for _ in range(length)]
            return value
        else:
            raise ValueError(f"Unknown tag type: {tag_type}")

    def read_list(self) -> List[Any]:
        tag_type = self.read_ubyte()
        length = self.read_int()
        return [self.read_tag(tag_type) for _ in range(length)]

    def read_compound(self) -> Dict[str, Any]:
        result = {}
        while True:
            tag_type = self.read_ubyte()
            if tag_type == self.TAG_END:
                break
            name = self.read_string()
            value = self.read_tag(tag_type)
            result[name] = value
        return result

    def parse(self) -> Dict[str, Any]:
        tag_type = self.read_ubyte()
        if tag_type != self.TAG_COMPOUND:
            raise ValueError(f"Expected compound tag, got {tag_type}")
        name = self.read_string()
        return {name: self.read_compound()}


class NBTWriter:
    """Write Minecraft NBT (Named Binary Tag) format"""

    def __init__(self):
        self.data = bytearray()

    def write_byte(self, value: int):
        self.data.extend(struct.pack('>b', value))

    def write_ubyte(self, value: int):
        self.data.extend(struct.pack('>B', value))

    def write_short(self, value: int):
        self.data.extend(struct.pack('>h', value))

    def write_int(self, value: int):
        self.data.extend(struct.pack('>i', value))

    def write_long(self, value: int):
        # Handle unsigned longs by converting to signed if needed
        if value > 0x7FFFFFFFFFFFFFFF:
            value = value - 0x10000000000000000
        self.data.extend(struct.pack('>q', value))

    def write_float(self, value: float):
        self.data.extend(struct.pack('>f', value))

    def write_double(self, value: float):
        self.data.extend(struct.pack('>d', value))

    def write_string(self, value: str):
        encoded = value.encode('utf-8')
        self.write_short(len(encoded))
        self.data.extend(encoded)

    def write_tag(self, value: Any, tag_type: int):
        if tag_type == NBTParser.TAG_BYTE:
            self.write_byte(value)
        elif tag_type == NBTParser.TAG_SHORT:
            self.write_short(value)
        elif tag_type == NBTParser.TAG_INT:
            self.write_int(value)
        elif tag_type == NBTParser.TAG_LONG:
            self.write_long(value)
        elif tag_type == NBTParser.TAG_FLOAT:
            self.write_float(value)
        elif tag_type == NBTParser.TAG_DOUBLE:
            self.write_double(value)
        elif tag_type == NBTParser.TAG_BYTE_ARRAY:
            self.write_int(len(value))
            for v in value:
                self.write_byte(v)
        elif tag_type == NBTParser.TAG_STRING:
            self.write_string(value)
        elif tag_type == NBTParser.TAG_LIST:
            self.write_list(value)
        elif tag_type == NBTParser.TAG_COMPOUND:
            self.write_compound(value)
        elif tag_type == NBTParser.TAG_INT_ARRAY:
            self.write_int(len(value))
            for v in value:
                self.write_int(v)
        elif tag_type == NBTParser.TAG_LONG_ARRAY:
            self.write_int(len(value))
            for v in value:
                self.write_long(v)

    def write_list(self, value: List[Any]):
        if len(value) == 0:
            self.write_ubyte(NBTParser.TAG_END)
            self.write_int(0)
            return

        # Infer tag type from first element
        first = value[0]
        if isinstance(first, bool) or isinstance(first, int) and -128 <= first <= 127:
            tag_type = NBTParser.TAG_BYTE
        elif isinstance(first, int) and -32768 <= first <= 32767:
            tag_type = NBTParser.TAG_SHORT
        elif isinstance(first, int):
            tag_type = NBTParser.TAG_INT
        elif isinstance(first, float):
            tag_type = NBTParser.TAG_DOUBLE
        elif isinstance(first, str):
            tag_type = NBTParser.TAG_STRING
        elif isinstance(first, list):
            tag_type = NBTParser.TAG_LIST
        elif isinstance(first, dict):
            tag_type = NBTParser.TAG_COMPOUND
        else:
            tag_type = NBTParser.TAG_COMPOUND

        self.write_ubyte(tag_type)
        self.write_int(len(value))
        for item in value:
            self.write_tag(item, tag_type)

    def write_compound(self, value: Dict[str, Any]):
        for name, item in value.items():
            # Infer tag type
            if isinstance(item, bool) or isinstance(item, int) and -128 <= item <= 127:
                tag_type = NBTParser.TAG_BYTE
            elif isinstance(item, int) and -32768 <= item <= 32767:
                tag_type = NBTParser.TAG_SHORT
            elif isinstance(item, int) and -2147483648 <= item <= 2147483647:
                tag_type = NBTParser.TAG_INT
            elif isinstance(item, int):
                tag_type = NBTParser.TAG_LONG
            elif isinstance(item, float):
                tag_type = NBTParser.TAG_DOUBLE
            elif isinstance(item, str):
                tag_type = NBTParser.TAG_STRING
            elif isinstance(item, list):
                if len(item) > 0 and all(isinstance(x, int) and -128 <= x <= 127 for x in item):
                    tag_type = NBTParser.TAG_BYTE_ARRAY
                elif len(item) > 0 and all(isinstance(x, int) and -2147483648 <= x <= 2147483647 for x in item):
                    tag_type = NBTParser.TAG_INT_ARRAY
                elif len(item) > 0 and all(isinstance(x, int) for x in item):
                    tag_type = NBTParser.TAG_LONG_ARRAY
                else:
                    tag_type = NBTParser.TAG_LIST
            elif isinstance(item, dict):
                tag_type = NBTParser.TAG_COMPOUND
            else:
                continue

            self.write_ubyte(tag_type)
            self.write_string(name)
            self.write_tag(item, tag_type)

        self.write_ubyte(NBTParser.TAG_END)

    def build(self, name: str, root: Dict[str, Any]) -> bytes:
        self.write_ubyte(NBTParser.TAG_COMPOUND)
        self.write_string(name)
        self.write_compound(root)
        return bytes(self.data)


def parse_litematic(file_path: str) -> Dict[str, Any]:
    """Parse a Litematica schematic file"""
    with gzip.open(file_path, 'rb') as f:
        data = f.read()

    parser = NBTParser(data)
    return parser.parse()


def write_litematic(file_path: str, data: Dict[str, Any]):
    """Write a Litematica schematic file"""
    writer = NBTWriter()
    root_name = list(data.keys())[0]
    root_data = data[root_name]
    nbt_data = writer.build(root_name, root_data)

    with gzip.open(file_path, 'wb') as f:
        f.write(nbt_data)


def rotate_point_2d(x: int, z: int, angle_degrees: int, pivot_x: int, pivot_z: int,
                   method: str = 'round') -> Tuple[int, int]:
    """Rotate a point in 2D around a pivot point

    Args:
        x, z: Point coordinates
        angle_degrees: Rotation angle
        pivot_x, pivot_z: Pivot point
        method: Rounding method - 'round' (default), 'floor', 'ceil', 'nearest_half'
    """
    # Translate to origin
    x -= pivot_x
    z -= pivot_z

    # Convert angle to radians
    angle_rad = math.radians(angle_degrees)

    # Rotate
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    new_x = x * cos_a - z * sin_a
    new_z = x * sin_a + z * cos_a

    # Translate back
    new_x = new_x + pivot_x
    new_z = new_z + pivot_z

    # Apply rounding method
    if method == 'floor':
        new_x = math.floor(new_x)
        new_z = math.floor(new_z)
    elif method == 'ceil':
        new_x = math.ceil(new_x)
        new_z = math.ceil(new_z)
    elif method == 'nearest_half':
        # Only place blocks if they're within 0.3 of an integer position (reduces dithering)
        x_frac = abs(new_x - round(new_x))
        z_frac = abs(new_z - round(new_z))
        if x_frac > 0.3 or z_frac > 0.3:
            # Use floor for positions far from integers to create smoother lines
            new_x = math.floor(new_x + 0.5)
            new_z = math.floor(new_z + 0.5)
        else:
            new_x = round(new_x)
            new_z = round(new_z)
    else:  # 'round' (default)
        new_x = round(new_x)
        new_z = round(new_z)

    return new_x, new_z


def rotate_facing(facing: str, angle_degrees: int) -> str:
    """Rotate a facing direction (north, south, east, west)"""
    # Normalize angle to 0-360
    angle = angle_degrees % 360

    # Define rotation map for 90-degree increments
    facings = ['north', 'east', 'south', 'west']

    if facing not in facings:
        return facing

    current_idx = facings.index(facing)
    rotation_steps = round(angle / 90) % 4
    new_idx = (current_idx + rotation_steps) % 4

    return facings[new_idx]


def rotate_axis(axis: str, angle_degrees: int) -> str:
    """Rotate an axis value (x, y, z)"""
    # Normalize angle to 0-360
    angle = angle_degrees % 360

    if axis == 'y':
        return 'y'  # Y axis doesn't change with horizontal rotation

    # For 90-degree rotations
    if abs(angle - 90) < 1:
        return 'z' if axis == 'x' else 'x'
    elif abs(angle - 180) < 1:
        return axis  # Same axis but direction might flip
    elif abs(angle - 270) < 1:
        return 'z' if axis == 'x' else 'x'

    return axis


def rotate_block_state(block_state: Dict[str, Any], angle_degrees: int) -> Dict[str, Any]:
    """Rotate a block state's directional properties"""
    result = block_state.copy()

    if 'Properties' in result:
        props = result['Properties'].copy()

        # Rotate facing property
        if 'facing' in props:
            props['facing'] = rotate_facing(props['facing'], angle_degrees)

        # Rotate axis property
        if 'axis' in props:
            props['axis'] = rotate_axis(props['axis'], angle_degrees)

        # Rotate rotation property (0-15 for blocks like signs)
        if 'rotation' in props:
            try:
                rotation = int(props['rotation'])
                # Each rotation value is 22.5 degrees (360/16)
                rotation_steps = round(angle_degrees / 22.5)
                new_rotation = (rotation + rotation_steps) % 16
                props['rotation'] = str(new_rotation)
            except:
                pass

        result['Properties'] = props

    return result


def extract_blocks_from_region(region_data: Dict[str, Any]) -> List[Tuple[int, int, int, Dict[str, Any]]]:
    """Extract all blocks from a region with their positions"""
    blocks = []

    palette = region_data.get('BlockStatePalette', [])
    block_states = region_data.get('BlockStates', [])
    size = region_data.get('Size', {})
    position = region_data.get('Position', {'x': 0, 'y': 0, 'z': 0})

    size_x = size.get('x', 0)
    size_y = size.get('y', 0)
    size_z = size.get('z', 0)

    if not palette or not block_states:
        return blocks

    # Calculate bits per block
    bits_per_block = max(2, (len(palette) - 1).bit_length())

    # Decode block states
    block_index = 0
    bit_pos = 0
    current_long_idx = 0

    for y in range(size_y):
        for z in range(size_z):
            for x in range(size_x):
                # Extract palette index
                palette_idx = 0
                bits_remaining = bits_per_block

                while bits_remaining > 0:
                    if current_long_idx >= len(block_states):
                        break

                    current_long = block_states[current_long_idx]
                    bits_available = 64 - bit_pos
                    bits_to_read = min(bits_remaining, bits_available)

                    mask = (1 << bits_to_read) - 1
                    value = (current_long >> bit_pos) & mask

                    palette_idx |= value << (bits_per_block - bits_remaining)

                    bit_pos += bits_to_read
                    bits_remaining -= bits_to_read

                    if bit_pos >= 64:
                        bit_pos = 0
                        current_long_idx += 1

                if palette_idx < len(palette):
                    block_state = palette[palette_idx]
                    # Only store non-air blocks
                    if block_state.get('Name', '') != 'minecraft:air':
                        abs_x = x + position['x']
                        abs_y = y + position['y']
                        abs_z = z + position['z']
                        blocks.append((abs_x, abs_y, abs_z, block_state))

    return blocks


def count_blocks_in_region(region_data: Dict[str, Any]) -> Dict[str, int]:
    """Count the number of each block type in a region"""
    block_counts = defaultdict(int)

    palette = region_data.get('BlockStatePalette', [])
    block_states = region_data.get('BlockStates', [])
    size = region_data.get('Size', {})

    size_x = size.get('x', 0)
    size_y = size.get('y', 0)
    size_z = size.get('z', 0)

    if not palette or not block_states:
        return block_counts

    # Calculate bits per block
    bits_per_block = max(2, (len(palette) - 1).bit_length())

    # Decode block states
    block_index = 0
    bit_pos = 0
    current_long_idx = 0

    for y in range(size_y):
        for z in range(size_z):
            for x in range(size_x):
                # Extract palette index
                palette_idx = 0
                bits_remaining = bits_per_block

                while bits_remaining > 0:
                    if current_long_idx >= len(block_states):
                        break

                    current_long = block_states[current_long_idx]
                    bits_available = 64 - bit_pos
                    bits_to_read = min(bits_remaining, bits_available)

                    mask = (1 << bits_to_read) - 1
                    value = (current_long >> bit_pos) & mask

                    palette_idx |= value << (bits_per_block - bits_remaining)

                    bit_pos += bits_to_read
                    bits_remaining -= bits_to_read

                    if bit_pos >= 64:
                        bit_pos = 0
                        current_long_idx += 1

                if palette_idx < len(palette):
                    block_state = palette[palette_idx]
                    block_name = block_state.get('Name', 'unknown')

                    # Create key, filtering out orientation-only properties
                    if 'Properties' in block_state and block_state['Properties']:
                        props = block_state['Properties']
                        # Filter out properties that are purely orientation (axis, facing, rotation)
                        # Keep properties that affect functionality (type, waterlogged, etc.)
                        filtered_props = {k: v for k, v in props.items()
                                         if k not in ['axis', 'facing', 'rotation']}

                        if filtered_props:
                            props_str = ','.join(f"{k}={v}" for k, v in sorted(filtered_props.items()))
                            block_key = f"{block_name}[{props_str}]"
                        else:
                            block_key = block_name
                    else:
                        block_key = block_name

                    block_counts[block_key] += 1

    return dict(block_counts)


def encode_blocks_to_region(blocks: List[Tuple[int, int, int, Dict[str, Any]]],
                            region_name: str = "Rotated") -> Dict[str, Any]:
    """Encode blocks into a region structure"""

    if not blocks:
        return {}

    # Calculate bounding box
    min_x = min(b[0] for b in blocks)
    max_x = max(b[0] for b in blocks)
    min_y = min(b[1] for b in blocks)
    max_y = max(b[1] for b in blocks)
    min_z = min(b[2] for b in blocks)
    max_z = max(b[2] for b in blocks)

    size_x = max_x - min_x + 1
    size_y = max_y - min_y + 1
    size_z = max_z - min_z + 1

    # Create palette and block array
    palette = [{'Name': 'minecraft:air'}]
    palette_map = {'minecraft:air': 0}

    # Initialize 3D array with air
    volume = size_x * size_y * size_z
    block_array = [0] * volume

    # Process blocks
    for x, y, z, block_state in blocks:
        # Normalize to local coordinates
        local_x = x - min_x
        local_y = y - min_y
        local_z = z - min_z

        # Create block state key
        block_key = str(block_state)

        # Add to palette if new
        if block_key not in palette_map:
            palette_map[block_key] = len(palette)
            palette.append(block_state)

        # Calculate index (XZY order as used by Litematica)
        idx = (local_y * size_z * size_x) + (local_z * size_x) + local_x
        block_array[idx] = palette_map[block_key]

    # Encode block array into long array
    bits_per_block = max(2, (len(palette) - 1).bit_length())
    longs_needed = ((volume * bits_per_block) + 63) // 64
    block_states_long = [0] * longs_needed

    bit_pos = 0
    long_idx = 0

    for palette_idx in block_array:
        bits_remaining = bits_per_block
        value = palette_idx

        while bits_remaining > 0:
            bits_available = 64 - bit_pos
            bits_to_write = min(bits_remaining, bits_available)

            mask = (1 << bits_to_write) - 1
            block_states_long[long_idx] |= ((value >> (bits_per_block - bits_remaining)) & mask) << bit_pos

            bit_pos += bits_to_write
            bits_remaining -= bits_to_write

            if bit_pos >= 64:
                bit_pos = 0
                long_idx += 1

    # Build region structure
    region = {
        'Position': {'x': min_x, 'y': min_y, 'z': min_z},
        'Size': {'x': size_x, 'y': size_y, 'z': size_z},
        'BlockStatePalette': palette,
        'BlockStates': block_states_long,
        'TileEntities': [],
        'PendingBlockTicks': [],
        'PendingFluidTicks': [],
        'Entities': []
    }

    return {region_name: region}


def rotate_schematic(data: Dict[str, Any], angle_degrees: int,
                    pivot_x: int = 0, pivot_z: int = 0, method: str = 'round') -> Dict[str, Any]:
    """Rotate an entire schematic

    Args:
        data: Schematic data
        angle_degrees: Rotation angle
        pivot_x, pivot_z: Pivot point
        method: Rotation method - 'round', 'floor', 'ceil', 'nearest_half'
    """

    root_name = list(data.keys())[0]
    root = data[root_name].copy()

    # Extract all blocks from all regions
    all_blocks = []

    if 'Regions' in root:
        for region_name, region_data in root['Regions'].items():
            blocks = extract_blocks_from_region(region_data)
            all_blocks.extend(blocks)

    print(f"Extracted {len(all_blocks)} blocks from schematic")

    # Rotate blocks
    rotated_blocks = []
    for x, y, z, block_state in all_blocks:
        new_x, new_z = rotate_point_2d(x, z, angle_degrees, pivot_x, pivot_z, method)
        new_block_state = rotate_block_state(block_state, angle_degrees)
        rotated_blocks.append((new_x, y, new_z, new_block_state))

    print(f"Rotated {len(rotated_blocks)} blocks by {angle_degrees}° using '{method}' method")

    # Encode back to region
    new_regions = encode_blocks_to_region(rotated_blocks, "Rotated")

    # Update metadata
    if 'Metadata' in root:
        metadata = root['Metadata'].copy()

        # Update name
        original_name = metadata.get('Name', 'Unnamed')
        metadata['Name'] = f"{original_name} (rotated {angle_degrees}°)"

        # Update timestamp
        metadata['TimeModified'] = int(time.time() * 1000)

        # Recalculate size
        if new_regions:
            region_data = list(new_regions.values())[0]
            size = region_data['Size']
            pos = region_data['Position']

            metadata['EnclosingSize'] = {
                'x': size['x'],
                'y': size['y'],
                'z': size['z']
            }

            metadata['TotalVolume'] = size['x'] * size['y'] * size['z']
            metadata['TotalBlocks'] = len(rotated_blocks)

        root['Metadata'] = metadata

    root['Regions'] = new_regions

    return {root_name: root}


def display_schematic_info(data: Dict[str, Any]):
    """Display information about the schematic"""

    # Get the root data
    root = data.get('', data)

    print("=" * 80)
    print("LITEMATICA SCHEMATIC FILE")
    print("=" * 80)
    print()

    # Metadata
    if 'Metadata' in root:
        metadata = root['Metadata']
        print("METADATA:")
        print("-" * 80)
        print(f"  Name: {metadata.get('Name', 'N/A')}")
        print(f"  Author: {metadata.get('Author', 'N/A')}")
        print(f"  Description: {metadata.get('Description', 'N/A')}")
        print(f"  Time Created: {metadata.get('TimeCreated', 'N/A')}")
        print(f"  Time Modified: {metadata.get('TimeModified', 'N/A')}")
        print(f"  Version: {metadata.get('Version', 'N/A')}")

        if 'EnclosingSize' in metadata:
            size = metadata['EnclosingSize']
            print(f"  Enclosing Size: {size.get('x', 0)} x {size.get('y', 0)} x {size.get('z', 0)}")

        if 'TotalVolume' in metadata:
            print(f"  Total Volume: {metadata['TotalVolume']:,} blocks")

        if 'TotalBlocks' in metadata:
            print(f"  Total Blocks: {metadata['TotalBlocks']:,}")

        if 'RegionCount' in metadata:
            print(f"  Region Count: {metadata['RegionCount']}")

        print()

    # Regions
    if 'Regions' in root:
        regions = root['Regions']
        print("REGIONS:")
        print("-" * 80)

        for region_name, region_data in regions.items():
            print(f"\n  Region: {region_name}")

            if 'Position' in region_data:
                pos = region_data['Position']
                print(f"    Position: ({pos.get('x', 0)}, {pos.get('y', 0)}, {pos.get('z', 0)})")

            if 'Size' in region_data:
                size = region_data['Size']
                print(f"    Size: {size.get('x', 0)} x {size.get('y', 0)} x {size.get('z', 0)}")
                volume = size.get('x', 0) * size.get('y', 0) * size.get('z', 0)
                print(f"    Volume: {volume:,} blocks")

                # Calculate and display center point
                if 'Position' in region_data:
                    pos = region_data['Position']
                    center_x = pos.get('x', 0) + size.get('x', 0) // 2
                    center_y = pos.get('y', 0) + size.get('y', 0) // 2
                    center_z = pos.get('z', 0) + size.get('z', 0) // 2
                    print(f"    Center Point: ({center_x}, {center_y}, {center_z})")

            # Block states and palette
            if 'BlockStatePalette' in region_data:
                palette = region_data['BlockStatePalette']

            # Count actual block usage
            print(f"\n    Block Usage:")
            block_usage = count_blocks_in_region(region_data)

            # Sort by count (descending) and then by name
            sorted_blocks = sorted(block_usage.items(), key=lambda x: (-x[1], x[0]))

            # Calculate total non-air blocks
            total_blocks = sum(count for name, count in sorted_blocks if 'air' not in name.lower())
            air_count = sum(count for name, count in sorted_blocks if 'air' in name.lower())

            # Count unique block types (excluding air)
            unique_non_air = len([name for name, count in sorted_blocks if 'air' not in name.lower()])

            print(f"      Total blocks: {total_blocks:,} ({unique_non_air} unique types)")
            if air_count > 0:
                print(f"      Air blocks: {air_count:,}")

            # Show non-air blocks
            non_air_blocks = [(name, count) for name, count in sorted_blocks if 'air' not in name.lower()]

            if non_air_blocks:
                print()
                for block_name, count in non_air_blocks[:30]:  # Show top 30
                    percentage = (count / total_blocks * 100) if total_blocks > 0 else 0
                    # Clean up block name for display
                    display_name = block_name.replace('minecraft:', '')
                    print(f"        {display_name:45s} {count:8,} ({percentage:5.2f}%)")

                if len(non_air_blocks) > 30:
                    remaining = len(non_air_blocks) - 30
                    remaining_count = sum(count for _, count in non_air_blocks[30:])
                    print(f"        ... and {remaining} more types ({remaining_count:,} blocks)")

            if 'BlockStates' in region_data:
                block_states = region_data['BlockStates']
                print(f"\n    Block States Data: {len(block_states)} longs")

            if 'TileEntities' in region_data:
                tile_entities = region_data['TileEntities']
                print(f"    Tile Entities: {len(tile_entities)}")
                for i, te in enumerate(tile_entities[:5]):  # Show first 5
                    te_id = te.get('Id', te.get('id', 'unknown'))
                    pos = (te.get('x', 0), te.get('y', 0), te.get('z', 0))
                    print(f"      {i+1}. {te_id} at {pos}")
                if len(tile_entities) > 5:
                    print(f"      ... and {len(tile_entities) - 5} more")

            if 'Entities' in region_data:
                entities = region_data['Entities']
                print(f"    Entities: {len(entities)}")

        print()

    print("=" * 80)


def plot_schematic_2d(data: Dict[str, Any], output_path: str = None, y_level: int = None, show: bool = True):
    """Create a 2D overhead plot of the schematic"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.colors import ListedColormap
    except ImportError:
        print("Error: matplotlib is required for plotting.")
        print("Install it with: pip install matplotlib")
        return False

    root = list(data.values())[0]

    # Extract all blocks from all regions
    all_blocks = []
    if 'Regions' in root:
        for region_name, region_data in root['Regions'].items():
            blocks = extract_blocks_from_region(region_data)
            all_blocks.extend(blocks)

    if not all_blocks:
        print("No blocks found to plot")
        return False

    print(f"Plotting {len(all_blocks)} blocks...")

    # Filter by Y level if specified
    if y_level is not None:
        all_blocks = [(x, y, z, block) for x, y, z, block in all_blocks if y == y_level]
        print(f"Filtered to {len(all_blocks)} blocks at Y={y_level}")

    # Get unique block types and assign colors
    block_types = {}
    for x, y, z, block_state in all_blocks:
        block_name = block_state.get('Name', 'unknown').replace('minecraft:', '')
        if block_name not in block_types:
            block_types[block_name] = len(block_types)

    # Create a color map
    colors = plt.cm.tab20c(range(len(block_types)))
    if len(block_types) > 20:
        colors = plt.cm.hsv([i / len(block_types) for i in range(len(block_types))])

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot blocks
    for x, y, z, block_state in all_blocks:
        block_name = block_state.get('Name', 'unknown').replace('minecraft:', '')
        color_idx = block_types[block_name]
        ax.scatter(x, z, c=[colors[color_idx]], s=1, marker='s')

    # Add legend for most common blocks (top 15)
    block_counts = defaultdict(int)
    for x, y, z, block_state in all_blocks:
        block_name = block_state.get('Name', 'unknown').replace('minecraft:', '')
        block_counts[block_name] += 1

    top_blocks = sorted(block_counts.items(), key=lambda x: -x[1])[:15]
    legend_elements = [mpatches.Patch(facecolor=colors[block_types[name]],
                                     label=f'{name} ({count})')
                      for name, count in top_blocks]

    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    # Labels and title
    ax.set_xlabel('X (East →)', fontsize=10)
    ax.set_ylabel('Z (South →)', fontsize=10)

    title = 'Litematica Schematic - Overhead View'
    if y_level is not None:
        title += f' (Y={y_level})'
    ax.set_title(title, fontsize=12, fontweight='bold')

    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")

    if show:
        plt.show()

    plt.close()
    return True


def cmd_info(args):
    """Handle the info subcommand"""
    # Resolve input file path
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = Path.cwd() / input_path

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    print(f"Loading schematic from: {input_path}")
    print()

    try:
        data = parse_litematic(str(input_path))
        display_schematic_info(data)
        return 0

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_rotate(args):
    """Handle the rotate subcommand"""
    # Resolve input file path
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = Path.cwd() / input_path

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    print(f"Loading schematic from: {input_path}")
    print()

    try:
        data = parse_litematic(str(input_path))

        print(f"Rotating by {args.angle}° around pivot point ({args.x}, {args.z})")
        print()

        rotated_data = rotate_schematic(data, args.angle, args.x, args.z, args.method)

        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = input_path.parent / f"{input_path.stem}_rotated.litematic"

        print()
        print(f"Writing rotated schematic to: {output_path}")
        write_litematic(str(output_path), rotated_data)
        print(f"Done! Rotated schematic saved.")
        print()

        # Display info about rotated schematic
        display_schematic_info(rotated_data)

        return 0

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_plot(args):
    """Handle the plot subcommand"""
    # Resolve input file path
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = Path.cwd() / input_path

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    print(f"Loading schematic from: {input_path}")
    print()

    try:
        data = parse_litematic(str(input_path))

        # Determine output path if saving
        output_path = None
        if args.output:
            output_path = Path(args.output)
        elif args.save:
            output_path = input_path.parent / f"{input_path.stem}_plot.png"

        # Create the plot
        success = plot_schematic_2d(data,
                                   output_path=str(output_path) if output_path else None,
                                   y_level=args.y_level,
                                   show=not args.no_show)

        return 0 if success else 1

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


def swap_blocks_in_schematic(data: Dict[str, Any], from_block: str, to_block: str) -> Dict[str, Any]:
    """Swap all instances of one block type with another"""

    # Normalize block names
    if not from_block.startswith('minecraft:'):
        from_block = f'minecraft:{from_block}'
    if not to_block.startswith('minecraft:'):
        to_block = f'minecraft:{to_block}'

    root_name = list(data.keys())[0]
    root = data[root_name].copy()

    swap_count = 0

    # Process each region
    if 'Regions' in root:
        new_regions = {}
        for region_name, region_data in root['Regions'].items():
            region_copy = region_data.copy()

            # Check and update palette
            if 'BlockStatePalette' in region_copy:
                palette = region_copy['BlockStatePalette']
                new_palette = []

                for block_state in palette:
                    block_name = block_state.get('Name', '')

                    if block_name == from_block:
                        # Create new block state with target block
                        new_block_state = {'Name': to_block}
                        # Preserve properties if compatible
                        if 'Properties' in block_state:
                            new_block_state['Properties'] = block_state['Properties'].copy()
                        new_palette.append(new_block_state)
                        swap_count += 1
                    else:
                        new_palette.append(block_state)

                region_copy['BlockStatePalette'] = new_palette

            new_regions[region_name] = region_copy

        root['Regions'] = new_regions

    # Update metadata
    if 'Metadata' in root:
        metadata = root['Metadata'].copy()
        metadata['TimeModified'] = int(time.time() * 1000)
        root['Metadata'] = metadata

    print(f"Swapped {swap_count} palette entries from {from_block} to {to_block}")

    return {root_name: root}


def cmd_swap(args):
    """Handle the swap subcommand"""
    # Resolve input file path
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = Path.cwd() / input_path

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    print(f"Loading schematic from: {input_path}")
    print()

    try:
        data = parse_litematic(str(input_path))

        print(f"Swapping blocks: {args.from_block} → {args.to_block}")
        print()

        swapped_data = swap_blocks_in_schematic(data, args.from_block, args.to_block)

        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = input_path.parent / f"{input_path.stem}_swapped.litematic"

        print()
        print(f"Writing modified schematic to: {output_path}")
        write_litematic(str(output_path), swapped_data)
        print(f"Done! Swapped schematic saved.")
        print()

        # Display info about swapped schematic
        display_schematic_info(swapped_data)

        return 0

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    parser = argparse.ArgumentParser(
        description='Parse and manipulate Minecraft Litematica schematic files',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Create subparsers
    subparsers = parser.add_subparsers(
        title='commands',
        description='Available commands',
        dest='command',
        help='Command to execute'
    )

    # Info subcommand
    info_parser = subparsers.add_parser(
        'info',
        help='Display schematic information',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Display detailed information about a Litematica schematic file',
        epilog="""
Examples:
  # Display schematic info
  %(prog)s structure.litematic
  %(prog)s -i my_structure.litematic
        """
    )
    info_parser.add_argument('input', nargs='?', help='Input litematic schematic file')
    info_parser.add_argument('-i', '--input-file', dest='input_alt', help='Input litematic schematic file (alternative syntax)')
    info_parser.set_defaults(func=cmd_info)

    # Rotate subcommand
    rotate_parser = subparsers.add_parser(
        'rotate',
        help='Rotate a schematic around a pivot point',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Rotate a Litematica schematic by an arbitrary angle around a specified pivot point',
        epilog="""
Examples:
  # Rotate 90 degrees clockwise around origin
  %(prog)s structure.litematic 90
  %(prog)s -i structure.litematic -r 90

  # Rotate 45 degrees around custom pivot point
  %(prog)s structure.litematic 45 -x 100 -z 200
  %(prog)s -i structure.litematic -r 45 -x 100 -z 200

  # Rotate -90 degrees (counter-clockwise)
  %(prog)s structure.litematic -90
  %(prog)s -i structure.litematic -r -90 -o output.litematic

Coordinate System:
  X axis = East (+) / West (-)
  Z axis = South (+) / North (-)
  Positive angles rotate clockwise when viewed from above

Rotation Methods:
  round        - Standard rounding (default, may cause dithering)
  floor        - Always round down (bias toward negative)
  ceil         - Always round up (bias toward positive)
  nearest_half - Smoother results, reduces dithering for angled rotations
        """
    )
    rotate_parser.add_argument('input', nargs='?', help='Input litematic schematic file')
    rotate_parser.add_argument('angle', nargs='?', type=int, help='Rotation angle in degrees')
    rotate_parser.add_argument('-i', '--input-file', dest='input_alt', help='Input litematic schematic file (alternative syntax)')
    rotate_parser.add_argument('-r', '--rotate', dest='angle_alt', type=int,
                              help='Rotation angle in degrees (positive = clockwise when viewed from above)')
    rotate_parser.add_argument('-x', type=int, default=0,
                              help='X coordinate of pivot point (default: 0)')
    rotate_parser.add_argument('-z', type=int, default=0,
                              help='Z coordinate of pivot point (default: 0)')
    rotate_parser.add_argument('-m', '--method', default='round',
                              choices=['round', 'floor', 'ceil', 'nearest_half'],
                              help='Rotation rounding method (default: round). Use "nearest_half" for smoother results.')
    rotate_parser.add_argument('-o', '--output', help='Output file path (default: <input>_rotated.litematic)')
    rotate_parser.set_defaults(func=cmd_rotate)

    # Plot subcommand
    plot_parser = subparsers.add_parser(
        'plot',
        help='Create a 2D overhead visualization of a schematic',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Generate a 2D bird\'s eye view plot of a Litematica schematic',
        epilog="""
Examples:
  # Display overhead view of schematic
  %(prog)s structure.litematic

  # Save plot to file
  %(prog)s structure.litematic -o plot.png
  %(prog)s structure.litematic --save

  # Show only specific Y level
  %(prog)s structure.litematic -y 4

  # Save without showing
  %(prog)s structure.litematic -o plot.png --no-show

Notes:
  - Requires matplotlib: pip install matplotlib
  - Each block type is shown in a different color
  - Legend shows the 15 most common blocks
        """
    )
    plot_parser.add_argument('input', nargs='?', help='Input litematic schematic file')
    plot_parser.add_argument('-i', '--input-file', dest='input_alt', help='Input litematic schematic file (alternative syntax)')
    plot_parser.add_argument('-o', '--output', help='Output image file path (PNG, JPG, etc.)')
    plot_parser.add_argument('--save', action='store_true', help='Save plot as <input>_plot.png')
    plot_parser.add_argument('-y', '--y-level', dest='y_level', type=int, help='Only show blocks at this Y level')
    plot_parser.add_argument('--no-show', action='store_true', help='Don\'t display the plot window (only save)')
    plot_parser.set_defaults(func=cmd_plot)

    # Swap subcommand
    swap_parser = subparsers.add_parser(
        'swap',
        help='Replace one block type with another',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Swap all instances of one block type with another in a Litematica schematic',
        epilog="""
Examples:
  # Swap stone for dirt
  %(prog)s structure.litematic stone dirt
  %(prog)s -i structure.litematic -f stone -t dirt

  # With full namespace
  %(prog)s structure.litematic minecraft:stone minecraft:dirt

  # Save with custom output
  %(prog)s structure.litematic stone dirt -o modified.litematic

  # Swap wool colors
  %(prog)s structure.litematic orange_wool yellow_wool

Notes:
  - Block names can be with or without 'minecraft:' namespace
  - Properties (like axis, facing) are preserved when compatible
  - The palette is updated, but block placement remains the same
        """
    )
    swap_parser.add_argument('input', nargs='?', help='Input litematic schematic file')
    swap_parser.add_argument('from_block', nargs='?', help='Block type to replace')
    swap_parser.add_argument('to_block', nargs='?', help='Block type to replace with')
    swap_parser.add_argument('-i', '--input-file', dest='input_alt', help='Input litematic schematic file (alternative syntax)')
    swap_parser.add_argument('-f', '--from', dest='from_block_alt', help='Block type to replace (alternative syntax)')
    swap_parser.add_argument('-t', '--to', dest='to_block_alt', help='Block type to replace with (alternative syntax)')
    swap_parser.add_argument('-o', '--output', help='Output file path (default: <input>_swapped.litematic)')
    swap_parser.set_defaults(func=cmd_swap)

    # Parse arguments
    args = parser.parse_args()

    # Check if a command was provided
    if not hasattr(args, 'func'):
        parser.print_help()
        return 1

    # Normalize arguments - handle both positional and flag syntax
    if hasattr(args, 'input_alt') and args.input_alt:
        args.input = args.input_alt

    if hasattr(args, 'angle_alt') and args.angle_alt:
        args.angle = args.angle_alt

    if hasattr(args, 'from_block_alt') and args.from_block_alt:
        args.from_block = args.from_block_alt

    if hasattr(args, 'to_block_alt') and args.to_block_alt:
        args.to_block = args.to_block_alt

    # Validate input
    if not args.input:
        print(f"Error: input file is required")
        return 1

    # For rotate command, validate angle
    if args.command == 'rotate' and args.angle is None:
        print(f"Error: rotation angle is required")
        rotate_parser.print_help()
        return 1

    # For swap command, validate blocks
    if args.command == 'swap':
        if not args.from_block:
            print(f"Error: source block type is required")
            swap_parser.print_help()
            return 1
        if not args.to_block:
            print(f"Error: target block type is required")
            swap_parser.print_help()
            return 1

    # Execute the appropriate command
    return args.func(args)


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
