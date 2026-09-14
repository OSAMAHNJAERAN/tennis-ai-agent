"""Review every accepted fragment link and the unlinked near-player boundary."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from PIL import Image, ImageDraw
from scripts.evaluate.audit_uvy_videos import digest
from src.utils.video_frame_sequence import VideoFrameSequence


def main():
    base = ROOT/'outputs/vision_upgrade_audit'
    folders = [base/'uvy_flow_fragment_linking_pilot01']
    dataset = ROOT/'data/external/uvy_tennis_videos'
    manifest = json.loads((dataset/'manifest.json').read_text())
    people_report = json.loads((base/'uvy_person_tracked640/report.json').read_text())
    for folder in folders:
        report = json.loads((folder/'report.json').read_text())
        if not report['complete'] or digest(dataset/'manifest.json') != report['dataset_manifest_sha256']:
            raise ValueError('Incomplete or changed source')
        cases = [(s, edge) for s, info in report['sequences'].items() for edge in info['links']]
        board = Image.new('RGB', (960, len(cases)*200), '#222222')
        painter = ImageDraw.Draw(board)
        for row, (sequence, edge) in enumerate(cases):
            info = people_report['sequences'][sequence]
            path = base/'uvy_person_tracked640'/info['predictions_file']
            if digest(path) != info['predictions_sha256']:
                raise ValueError('Person cache changed')
            people = json.loads(path.read_text())
            with VideoFrameSequence(str(dataset/manifest['sequences'][sequence]['video'])) as frames:
                for col, (identity, index) in enumerate(((edge['predecessor'], edge['end_frame']), (edge['successor'], edge['start_frame']))):
                    image = Image.fromarray(frames[index][:, :, ::-1])
                    box = next(r['box'] for r in people[index] if r['id'] == identity)
                    width, height = box[2]-box[0], box[3]-box[1]
                    x, y = (box[0]+box[2])/2, (box[1]+box[3])/2
                    radius = max(65, height*.7, width*.7)
                    left, top = max(0, int(x-radius)), max(0, int(y-radius))
                    right, bottom = min(image.width, int(x+radius)), min(image.height, int(y+radius))
                    d = ImageDraw.Draw(image)
                    d.rectangle(box, outline='#00ffff', width=2)
                    flow = edge['forward'] if col == 0 else edge['backward']
                    for point in flow.get('source_points', []):
                        px, py = point
                        d.ellipse((px-1, py-1, px+1, py+1), fill='#ffff00')
                    image = image.crop((left, top, right, bottom))
                    image.thumbnail((480, 170))
                    board.paste(image, (col*480, row*200+25))
                    status = 'REJECTED' if edge.get('rejected_target') else 'ACCEPTED'
                    painter.text((col*480+5, row*200+6), f'{status} {sequence} ID {identity}, frame {index+1}', fill='white')
        output = folder/'boundary_review.jpg'
        board.save(output, quality=70)
        (folder/'boundary_review.json').write_text(json.dumps({'report_sha256': digest(folder/'report.json'),
            'script_sha256': digest(__file__), 'image_sha256': digest(output), 'cases': cases,
            'scope': 'CROPPED_BOUNDARY_IMAGE_REVIEW; NOT_INDEPENDENT_IDENTITY_GROUND_TRUTH'}, indent=2), encoding='utf-8')
        print(str(output))


if __name__ == '__main__':
    main()
