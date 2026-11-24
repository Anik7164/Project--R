"""
How a phone finds this PC (the iPhone app works the ARGUS way):

- Bonjour: the server announces "_projectr._tcp" on the Wi-Fi with its
  addresses in the TXT record (hosts, port, path, name, v). The app lists
  the PCs it hears. Announced again when the addresses change.
- A pairing link, shown as a QR code on the PC (Connect a phone):
      projectr://connect?h=192.168.1.10,10.0.0.5&p=8000&k=<operator key>&n=<PC name>
  The iPhone camera opens it in the app; the app's own scanner reads it too.
- Typing the address, for Wi-Fi that blocks Bonjour (universities, hotels).

A PC can have several addresses (Wi-Fi, Ethernet, hotspot), so the phone
always gets the whole list, likely ones first, and tries them in turn.

Bonjour needs `pip install zeroconf`; without it the QR code and typing
still work.
"""

import ipaddress
import socket
import threading
import urllib.parse

SERVICE = "_projectr._tcp.local."
VIRTUAL = ("vethernet", "virtualbox", "vmware", "wsl", "hyper-v", "docker", "loopback",
           "tailscale", "zerotier", "vpn", "bluetooth", "vbox", "npcap", "tap", "tun")


def pc_name():
    name = socket.gethostname().split(".")[0].strip()
    return name or "Project R PC"


def primary_address():
    """The address of the way out (the Wi-Fi or Ethernet in use)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("10.255.255.255", 1))
            return probe.getsockname()[0]
    except OSError:
        return None


def lan_addresses():
    """This PC's private IPv4 addresses: the one in use first, virtual
    adapters (WSL, VirtualBox, VPNs) last."""
    first = primary_address()
    found = []
    try:
        import psutil

        stats = psutil.net_if_stats()
        for adapter, entries in psutil.net_if_addrs().items():
            if adapter in stats and not stats[adapter].isup:
                continue
            for entry in entries:
                if entry.family != socket.AF_INET:
                    continue
                try:
                    ip = ipaddress.ip_address(entry.address)
                except ValueError:
                    continue
                if not ip.is_private or ip.is_loopback or ip.is_link_local:
                    continue
                virtual = any(word in adapter.lower() for word in VIRTUAL)
                found.append((virtual, entry.address != first, entry.address))
    except Exception:          # psutil missing or the system refuses: the one we know
        pass
    addresses = []
    for _, _, address in sorted(found):
        if address not in addresses:
            addresses.append(address)
    if first and first not in addresses and not first.startswith("127."):
        addresses.insert(0, first)
    return addresses


def pairing_link(addresses, port, key="", name=None):
    query = {"h": ",".join(addresses), "p": str(port), "n": name or pc_name()}
    if key:
        query["k"] = key
    return "projectr://connect?" + urllib.parse.urlencode(query, safe=",")


class Announcer:
    """Keeps "_projectr._tcp" announced on the Wi-Fi while the server runs."""

    EVERY = 10.0

    def __init__(self, port):
        self.port = port
        self.name = pc_name()
        self.addresses = []
        self.error = None
        self._zeroconf = None
        self._info = None
        self._stop = threading.Event()

    def start(self):
        try:
            import zeroconf  # noqa: F401
        except ImportError:
            self.error = "pip install zeroconf (phones then find this PC by themselves)"
            return False
        threading.Thread(target=self._run, name="bonjour", daemon=True).start()
        return True

    def stop(self):
        self._stop.set()

    def _info_for(self, addresses):
        from zeroconf import ServiceInfo

        properties = {"hosts": ",".join(addresses), "port": str(self.port), "path": "/",
                      "name": self.name, "v": "1"}
        return ServiceInfo(SERVICE, f"{self.name}.{SERVICE}",
                           addresses=[socket.inet_aton(a) for a in addresses],
                           port=self.port, properties=properties,
                           server=f"{self.name.replace(' ', '-')}.local.")

    def _run(self):
        from zeroconf import IPVersion, Zeroconf

        try:
            self._zeroconf = Zeroconf(ip_version=IPVersion.V4Only)
            while not self._stop.is_set():
                addresses = lan_addresses()
                if addresses and addresses != self.addresses:
                    info = self._info_for(addresses)
                    if self._info is None:
                        self._zeroconf.register_service(info, allow_name_change=True)
                    else:
                        info.name = self._info.name     # the name it was given
                        self._zeroconf.update_service(info)
                    self._info, self.addresses = info, addresses
                self._stop.wait(self.EVERY)
        except Exception as error:      # no network yet, a firewall: QR and typing still work
            self.error = f"{type(error).__name__}: {error}"
        finally:
            if self._zeroconf is not None:
                try:
                    if self._info is not None:
                        self._zeroconf.unregister_service(self._info)
                    self._zeroconf.close()
                except Exception:
                    pass
