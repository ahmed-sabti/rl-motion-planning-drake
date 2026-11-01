import argparse
import os
import signal
import subprocess
import sys
import threading


def main():
    parser = argparse.ArgumentParser(description="Serve both robot dashboards in one process tree.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--meshcat-port", type=int, default=7800)
    parser.add_argument("--two-link-meshcat-port", type=int, default=7840)
    args = parser.parse_args()
    ports = (args.meshcat_port, args.two_link_meshcat_port)
    if any(not 1024 <= port <= 65515 for port in ports) or abs(ports[0] - ports[1]) <= 20:
        parser.error("Choose two non-overlapping Meshcat pools of 21 ports between 1024 and 65535.")
    if any(start <= port <= start + 20 for start in ports for port in (4880, 4881)):
        parser.error("Meshcat pools must not include dashboard ports 4880 or 4881.")
    stopping = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopping.set())
    children = []
    code = 0
    try:
        for variant, port, meshcat_port, peer in (
            ("3r", 4880, ports[0], "DASHBOARD_2R_PORT"),
            ("2r", 4881, ports[1], "DASHBOARD_PORT"),
        ):
            env = dict(os.environ, PLANAR2D_ROBOT=variant,
                       DASHBOARD_PEER_PORT=os.environ.get(peer, "4881" if variant == "3r" else "4880"),
                       DASHBOARD_PEER_LABEL="2-link arm" if variant == "3r" else "3-link arm")
            children.append(subprocess.Popen(
                [sys.executable, "-m", "dashboard.physics_dashboard", "--host", args.host,
                 "--port", str(port), "--meshcat-port", str(meshcat_port)],
                env=env, start_new_session=True))
        while not stopping.wait(.2):
            exited = next((child for child in children if child.poll() is not None), None)
            if exited is not None:
                code = exited.returncode or 1
                print("A dashboard exited; stopping the container.", file=sys.stderr, flush=True)
                break
    finally:
        for child in children:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
    return code


if __name__ == "__main__":
    sys.exit(main())
