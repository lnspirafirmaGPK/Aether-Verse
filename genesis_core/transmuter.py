import torch

class AssetTransmuter:
    def __init__(self):
        self.intent_labels = {0: "AGGRESSION", 1: "RESISTANCE", 2: "GENESIS"}

    def transmute(self, intent_probs, karma_score_tensor):
        # แปลง Tensor เป็นค่าที่อ่านได้
        intent_idx = torch.argmax(intent_probs).item()
        intent = self.intent_labels[intent_idx]
        score = torch.sigmoid(karma_score_tensor).item()
        
        # [span_11](start_span)โครงสร้าง JSON Object ที่จะส่งออกไปสร้าง Invoice/Ticket[span_11](end_span)
        asset_package = {
            "analysis_id": "GEN-" + str(int(torch.randint(1000, 9999, (1,)))),
            "detected_intent": intent,
            "karmic_score": round(score, 2),
            "commercial_asset": {},
            "fairness_audit": ""
        }
        
        # Logic การสร้างรายได้ตาม Blueprint
        if intent == "AGGRESSION":
            asset_package["commercial_asset"] = {
                "sku": "SRV-MAINT-L3",
                "name": "Emergency Cooling Protocol",
                "price_thb": 1500,
                "reason": "System Overheat Detected (High Resource Usage)"
            }
            asset_package["fairness_audit"] = "Server resources strictly allocated based on fair usage policy."

        elif intent == "RESISTANCE":
            asset_package["commercial_asset"] = {
                "sku": "RPT-OPT-BW",
                "name": "Infrastructure Optimization Report",
                "price_thb": 3500,
                "reason": "Data Bottleneck Identified"
            }
            asset_package["fairness_audit"] = "Latency caused by client-side data structure, not server load."

        elif intent == "GENESIS":
            asset_package["commercial_asset"] = {
                "sku": "MODEL-BP-V1",
                "name": "Premium Optimization Blueprint",
                "price_thb": 9900,
                "reason": "High Efficiency Pattern Captured"
            }
            asset_package["fairness_audit"] = "System operating at peak efficiency. This state is crystallizable."
            
        return asset_package
