# scripts/sentinel_bridge.py
import sys
import os
import torch
import json

# เพิ่ม path ให้ Python มองเห็น folder genesis_core
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from genesis_core.cortex import BehavioralCortex
from genesis_core.transmuter import AssetTransmuter

def run_sentinel():
    # 1. จำลองข้อมูล (ในอนาคตส่วนนี้จะดึงจาก Log จริงหรือไฟล์ภาพ Infrared)
    # [Batch=1, Time=10 sequences, Features=128]
    dummy_input = torch.randn(1, 10, 128)

    # 2. ปลุกสมอง (Instantiate Cortex)
    brain = BehavioralCortex()
    transmuter = AssetTransmuter()

    # 3. ให้ AI คิด (Forward Pass)
    intent_probs, karma_score = brain(dummy_input)

    # 4. แปลงความคิดเป็นเงิน (Transmute to Asset)
    # นี่คือกฎธุรกิจจาก transmuter.py ที่คุณต้องการ
    asset_data = transmuter.transmute(intent_probs, karma_score)

    # 5. ปริ้นท์ JSON ออกมาเพื่อให้ GitHub Actions อ่านค่า
    print(json.dumps(asset_data, indent=2))

if __name__ == "__main__":
    run_sentinel()
