"""Simple enclosure guide using RLViser 0.9.3's official AddRender protocol.

Coordinates follow RocketSim. This wire grid is visual only; the simulator's
rounded collision mesh remains authoritative. Goal mouths are left open.
Schema: https://github.com/VirxEC/rlviser/tree/v0.9.3/spec
"""
import socket
import struct


def arena_lines() -> list:
    lines = []
    wall = (0.65, 0.85, 1.0, 0.55)
    roof = (0.55, 0.75, 0.95, 0.32)
    def line(a, b, color=wall):
        lines.append((a, b, color))
    for x in (-4096, 4096):
        for y in range(-5120, 5121, 512):
            line((x, y, 0), (x, y, 2044))
        for z in (0, 256, 512, 768, 1024, 1280, 1536, 1792, 2044):
            line((x, -5120, z), (x, 5120, z))
    for y in (-5120, 5120):
        for x in range(-4096, 4097, 512):
            line((x, y, 642 if abs(x) < 893 else 0), (x, y, 2044))
        for z in (0, 256, 512, 768, 1024, 1280, 1536, 1792, 2044):
            if z < 642:
                line((-4096, y, z), (-893, y, z))
                line((893, y, z), (4096, y, z))
            else:
                line((-4096, y, z), (4096, y, z))
        color = (0.2, 0.55, 1.0, 1.0) if y < 0 else (1.0, 0.45, 0.1, 1.0)
        line((-893, y, 0), (-893, y, 642), color)
        line((-893, y, 642), (893, y, 642), color)
        line((893, y, 642), (893, y, 0), color)
    for x in range(-4096, 4097, 512):
        line((x, -5120, 2044), (x, 5120, 2044), roof)
    for y in range(-5120, 5121, 512):
        line((-4096, y, 2044), (4096, y, 2044), roof)
    return lines


def render_packet(lines: list, group: int) -> bytes:
    import flatbuffers
    builder = flatbuffers.Builder(8192)
    def inline(values):
        builder.Prep(4, 4 * len(values))
        for value in reversed(values):
            builder.PrependFloat32(value)
        return builder.Offset()
    commands = []
    for start, end, color in lines:
        builder.StartObject(3)
        builder.PrependStructSlot(2, inline(color), 0)
        builder.PrependStructSlot(1, inline(end), 0)
        builder.PrependStructSlot(0, inline(start), 0)
        commands.append(builder.EndObject())
    builder.StartVector(4, len(commands), 4)
    for offset in reversed(commands):
        builder.PrependUOffsetTRelative(offset)
    values = builder.EndVector()
    types = builder.CreateByteVector(bytes([2] * len(commands))) # Render.Line3D
    builder.StartObject(3)
    builder.PrependInt32Slot(0, group, 0)
    builder.PrependUOffsetTRelativeSlot(1, types, 0)
    builder.PrependUOffsetTRelativeSlot(2, values, 0)
    add_render = builder.EndObject()
    builder.StartObject(2)
    builder.PrependUint8Slot(0, 6, 0) # Message.AddRender
    builder.PrependUOffsetTRelativeSlot(1, add_render, 0)
    packet = builder.EndObject()
    builder.Finish(packet)
    data = bytes(builder.Output())
    return struct.pack(">Q", len(data)) + data


class ArenaOverlay:
    def __init__(self):
        lines = arena_lines()
        self.packets = [render_packet(lines[i:i + 48], 700 + i // 48)
                        for i in range(0, len(lines), 48)]
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self) -> None:
        for packet in self.packets:
            self.socket.sendto(packet, ("127.0.0.1", 45243))

    def close(self) -> None:
        self.socket.close()
