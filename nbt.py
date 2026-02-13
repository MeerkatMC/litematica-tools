#!/usr/bin/env python3
"""
Minecraft NBT (Named Binary Tag) format parser and writer
"""
import struct
from typing import Any, Dict, List

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
