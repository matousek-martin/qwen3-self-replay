"""Small Runpod v2 client; credentials never leave this control server."""
import argparse
import json
from pathlib import Path
import shlex
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "lab/pod.json"


def api_key():
    key = None
    for line in (ROOT / ".env").read_text().splitlines():
        line = line.strip().removeprefix("export ")
        if line.startswith("RUNPOD_API_KEY="):
            key = shlex.split(line.split("=", 1)[1], comments=True)[0]
    if not key:
        raise RuntimeError("RUNPOD_API_KEY missing")
    return key


def account():
    """Read only the live credit and hourly spend; never expose credentials."""
    request = urllib.request.Request('https://api.runpod.io/graphql',
        data=json.dumps({'query': 'query { myself { clientBalance currentSpendPerHr } }'}).encode(),
        headers={'Authorization': 'Bearer '+api_key(), 'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result=json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f'Runpod account HTTP {error.code}') from None
    if result.get('errors') or not result.get('data',{}).get('myself'):
        raise RuntimeError('Runpod account response unavailable')
    return result['data']['myself']


def api(path, method="GET", payload=None):
    key = api_key()
    request = urllib.request.Request(
        "https://api.runpod.io/v2/" + path,
        data=None if payload is None else json.dumps(payload).encode(),
        method=method,
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace").replace(key, "[REDACTED]")
        raise RuntimeError(f"Runpod HTTP {error.code}: {detail[:1500]}") from None


def quote(cloud):
    return api("catalog/gpus/" + urllib.parse.quote("NVIDIA GeForce RTX 4090")
               + "?include=AVAILABILITY&product=POD&count=1&minCudaVersion=12.8&cloud=" + cloud)


def summary(pod):
    return {key: pod.get(key) for key in (
        "id", "name", "status", "actions", "cost", "cloud", "dataCenterId",
        "cudaVersion", "createdAt", "startedAt", "ssh", "gpu", "image", "disk", "mounts")}


def ssh_args(pod):
    endpoint = (pod.get("ssh") or {}).get("direct")
    if not endpoint:
        raise RuntimeError("Direct SSH is not ready")
    return ["ssh", "-i", "/root/.ssh/thinkingcap_gpu", "-p", str(endpoint["port"]),
            "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=accept-new",
            "-o", "UserKnownHostsFile=" + str(ROOT / "lab/known_hosts"),
            endpoint["username"] + "@" + endpoint["host"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["quote", "status", "stop"])
    parser.add_argument("--id")
    args = parser.parse_args()
    if args.action == "quote":
        for cloud in ("COMMUNITY", "SECURE"):
            print(cloud, json.dumps(quote(cloud)))
        return
    pod_id = args.id or json.loads(STATE.read_text())["id"]
    pod = api("pods/" + pod_id)
    if args.action == "stop" and "stop" in pod.get("actions", []):
        pod = api("pods/" + pod_id + "/action", "POST", {"action": "stop"})
    print(json.dumps(summary(pod), indent=2))


if __name__ == "__main__":
    main()
