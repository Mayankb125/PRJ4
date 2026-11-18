import asyncio
from bleak import BleakScanner

async def main():
    print("Scanning for all nearby BLE devices for 5 seconds...")
    devices = await BleakScanner.discover(timeout=5.0, return_adv=True)
    print(f"\nDiscovered {len(devices)} device(s):")
    for address, (device, adv_data) in devices.items():
        name = device.name or adv_data.local_name or "<Unknown/Unnamed>"
        print(f"  • {name:30s} | MAC: {address} | RSSI: {adv_data.rssi} dBm")

if __name__ == "__main__":
    asyncio.run(main())
