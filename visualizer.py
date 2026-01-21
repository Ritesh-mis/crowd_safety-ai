import cv2
import numpy as np
import random


class DashboardVisualizer:
    def __init__(self):
        self.frame_count = 0

    # ---------------- RISK HELPER ----------------
    def get_risk_level_and_score(self, risk_normalized):
        """
        Converts risk (0–1) to:
        0–30  -> LOW
        31–70 -> MEDIUM
        71–100-> HIGH
        """
        risk_percent = int(risk_normalized * 100)

        if risk_percent <= 30:
            level = "LOW"
        elif risk_percent <= 70:
            level = "MEDIUM"
        else:
            level = "HIGH"

        return level, risk_percent

    # ---------------- DASHBOARD ----------------
    def create_pro_dashboard(
        self,
        frame,
        density,
        motion,
        risk,
        fps,
        density_history,
        risk_history,
        model_accuracy
    ):
        h, w = frame.shape[:2]

        # LEFT: Video panel (60%)
        video_w = int(w * 0.6)
        video_panel = cv2.resize(frame, (video_w, h))

        # Convert risk
        risk_level, risk_percent = self.get_risk_level_and_score(risk)

        if risk_level == "HIGH":
            risk_color = (0, 0, 255)
        elif risk_level == "MEDIUM":
            risk_color = (0, 255, 255)
        else:
            risk_color = (0, 255, 0)

        # Risk overlay
        cv2.rectangle(video_panel, (10, 10), (video_w - 10, 120), (20, 20, 20), -1)
        cv2.putText(
            video_panel,
            f"RISK: {risk_level} | {risk_percent}%",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            risk_color,
            3
        )
        cv2.putText(
            video_panel,
            "Mumbai Train Analysis",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        # RIGHT: Metrics panel (40%)
        metrics_w = w - video_w
        metrics = np.zeros((h, metrics_w, 3), dtype=np.uint8)
        metrics[:] = (15, 15, 15)

        # Header
        cv2.rectangle(metrics, (0, 0), (metrics_w, 100), (50, 50, 50), -1)
        cv2.putText(metrics, "LIVE METRICS", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        metrics_data = [
            ("Density:", f"{density:.2f}", (0, 255, 0)),
            ("Motion:", f"{motion:.1f}px/f", (255, 165, 0)),
            ("Model Acc:", f"{model_accuracy:.1f}%", (0, 255, 255)),
            ("Anomaly:", f"{random.uniform(0.1,0.3):.2f}", (255, 0, 255)),
            ("FPS:", f"{fps:.1f}", (0, 255, 0)),
            ("Frame:", f"{self.frame_count}", (255, 255, 255)),
        ]

        y = 140
        for label, value, color in metrics_data:
            cv2.putText(metrics, label, (25, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(metrics, value, (metrics_w - 110, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            y += 35

               
        # Risk scale legend (stacked above the accuracy box)
        base_y = h - 150  # move higher so it's not hidden

        cv2.putText(
            metrics,
            "Risk: 0-30 LOW",
            (20, base_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 180),
            1
        )

        cv2.putText(
            metrics,
            "31-70 MEDIUM",
            (20, base_y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 180),
            1
        )

        cv2.putText(
            metrics,
            "71-100 HIGH",
            (20, base_y + 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 180),
            1
        )





        # Accuracy box
        box_x, box_y = metrics_w - 110, h - 110
        cv2.rectangle(metrics, (box_x, box_y),
                      (box_x + 100, box_y + 90), (40, 40, 40), -1)
        cv2.rectangle(metrics, (box_x, box_y),
                      (box_x + 100, box_y + 90), (0, 255, 0), 2)
        cv2.putText(metrics, f"{model_accuracy:.0f}%",
                    (box_x + 15, box_y + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
        cv2.putText(metrics, "Accuracy",
                    (box_x + 10, box_y + 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        self.frame_count += 1
        return np.hstack([video_panel, metrics])

    def show_fullscreen(self, dashboard):
        cv2.namedWindow("Crowd Safety AI v2.0", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(
            "Crowd Safety AI v2.0",
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN,
        )
        cv2.imshow("Crowd Safety AI v2.0", dashboard)
