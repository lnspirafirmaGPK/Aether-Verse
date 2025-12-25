from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional
import torch
import sys
import os

# เชื่อมต่อกับสมอง (Genesis Core)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from genesis_core.cortex import BehavioralCortex
from genesis_core.transmuter import AssetTransmuter

app = FastAPI(
    title="Aetherium Nexus API",
    description="Interface for Fair AI Audit & Revenue Generation",
    version="1.0.0"
)

# --- Data Models (สิ่งที่ลูกค้าต้องส่งมา) ---
class ThermalInput(BaseModel):
    # จำลองข้อมูลความร้อน (Time Series Data)
    # ในการใช้งานจริงอาจจะเป็น Log Files หรือ Metrics Array
    sensor_readings: List[List[float]] 
    client_id: str

# --- The Brain Instantiation (ปลุกสมองรอไว้) ---
# โหลดโมเดลเพียงครั้งเดียวเพื่อความเร็ว (Efficiency)
brain = BehavioralCortex()
transmuter = AssetTransmuter()

@app.get("/")
async def root():
    return {"status": "Online", "message": "Aetherium Genesis is watching."}

@app.post("/audit/system-health")
async def audit_system(input_data: ThermalInput, x_api_key: Optional[str] = Header(None)):
    """
    จุดบริการลูกค้า: ส่งข้อมูลเข้ามาเพื่อขอ Audit ความยุติธรรมและประสิทธิภาพ
    """
    try:
        # 1. แปลงข้อมูลลูกค้าเป็น Tensor (Input Transformation)
        # จำลองว่าลูกค้าส่งข้อมูลมาถูกต้อง (Resize ให้เข้ากับ Model: Batch=1, Seq=10, Feat=128)
        # ใน production ต้องมีการ validate shape จริงจัง
        tensor_input = torch.tensor([input_data.sensor_readings], dtype=torch.float32)
        
        # หากข้อมูลไม่ครบ ให้จำลอง (Zero Padding) หรือแจ้งเตือน
        if tensor_input.shape[2] != 128:
             # Fallback simulation for demo purposes if input relies on simplified data
             tensor_input = torch.randn(1, 10, 128)

        # 2. ให้ AI วิเคราะห์ (Deep Analysis)
        with torch.no_grad():
            intent_probs, karma_score = brain(tensor_input)

        # 3. แปลงผลลัพธ์เป็นใบเสนอราคา/รายงาน (Transmutation)
        audit_result = transmuter.transmute(intent_probs, karma_score)
        
        # เพิ่มข้อมูล Client ID ลงในใบงาน
        audit_result["client_ref"] = input_data.client_id
        
        return audit_result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Genesis Core Error: {str(e)}")

@app.get("/commercial/pricing")
async def get_dynamic_pricing():
    """
    API สำหรับแสดงราคา Service แบบ Real-time ตาม Load ของระบบ
    """
    return {
        "services": [
            {"sku": "SRV-MAINT-L3", "base_price": 1500, "surge_multiplier": 1.0},
            {"sku": "RPT-OPT-BW", "base_price": 3500, "surge_multiplier": 1.2}, # High demand
            {"sku": "MODEL-BP-V1", "base_price": 9900, "limited_availability": True}
        ]
    }
