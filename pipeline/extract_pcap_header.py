import pyshark
import os
def extract_headers_only(pcap_path, output_txt_path):
    # Load the pcap file without including raw data
    cap = pyshark.FileCapture(
        pcap_path,
        use_json=True,
        include_raw=False
    )

    with open(output_txt_path, 'w') as out_file:
        for idx, packet in enumerate(cap):
            out_file.write(f"\n========== Packet #{idx + 1} ==========\n")
            out_file.write(f"Arrival Time: {packet.sniff_time}\n")

            for layer in packet.layers:
                # Skip the 'data' layer to exclude payload
                if layer.layer_name.lower() == 'data':
                    continue

                out_file.write(f"\n-- {layer.layer_name.upper()} Layer --\n")
                for field_name in getattr(layer, 'field_names', []):
                    # Skip fields that may contain payload data
                    if 'payload' in field_name.lower():
                        continue
                    try:
                        value = getattr(layer, field_name, None)
                        if not callable(value) and value is not None:
                            out_file.write(f"{field_name}: {value}\n")
                    except Exception as e:
                        out_file.write(f"{field_name}: <Error - {e}>\n")

            out_file.write("=" * 40 + "\n")

    cap.close()


def batch_process_pcap_folder(base_folder="attack_data"):
    for label_dir in ["0", "1"]:
        label_path = os.path.join(base_folder, label_dir)
        if not os.path.isdir(label_path):
            print(f"Warning: {label_path} is not a valid directory.")
            continue

        label_name = "attack" if label_dir == "0" else "benign"

        for fname in os.listdir(label_path):
            if not fname.lower().endswith(".pcap"):
                continue

            pcap_path = os.path.join(label_path, fname)
            out_filename = f"packet_{os.path.splitext(fname)[0]}_{label_name}.txt"
            out_path = os.path.join(label_path, out_filename)

            print(f"Processing: {pcap_path} → {out_path}")
            extract_headers_only(pcap_path, out_path)
