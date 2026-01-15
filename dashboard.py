import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from collections import deque
import time


class CrowdSafetyDashboard:
    def __init__(self, max_history=100):

        self.max_history = max_history

        # Historical Data
        self.density_history = deque(maxlen=max_history)
        self.risk_history = deque(maxlen=max_history)
        self.motion_history = deque(maxlen=max_history)
        self.anomaly_history = deque(maxlen=max_history)
        self.fps_history = deque(maxlen=max_history)
        self.timestamps = deque(maxlen=max_history)

        # Statistics
        self.total_frames = 0
        self.high_risk_frames = 0
        self.anomaly_count = 0
        self.start_time = time.time()

        # Alerts
        self.current_alerts = []
        self.alert_history = deque(maxlen=10)

        # Cached (for performance)
        self.cached_density_chart = None
        self.cached_risk_chart = None
        self.cached_motion_chart = None
        self.cached_risk_gauge = None
        self.cached_density_gauge = None
        self.cached_stat_panel = None
        self.cached_alert_panel = None
        self.cached_heatmap_legend = None


    def update_metrics(self, density, risk_level, motion_magnitude, anomaly_detected, fps):
        self.total_frames += 1
        current_time = time.time() - self.start_time

        self.density_history.append(density)
        self.risk_history.append(risk_level)
        self.motion_history.append(motion_magnitude)
        self.anomaly_history.append(1 if anomaly_detected else 0)
        self.fps_history.append(fps)
        self.timestamps.append(current_time)

        if risk_level > 0.7:
            self.high_risk_frames += 1
        if anomaly_detected:
            self.anomaly_count += 1


    def add_alert(self, alert_type, severity, message):
        alert = {
            "type": alert_type,
            "severity": severity,
            "message": message,
            "time": time.time() - self.start_time
        }
        self.current_alerts.append(alert)
        self.alert_history.append(alert)


    def clear_old_alerts(self, max_age=5.0):
        current_time = time.time() - self.start_time
        self.current_alerts = [a for a in self.current_alerts if current_time - a['time'] < max_age]


    def create_line_chart(self, data, title, color, ylabel, width, height):
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
        fig.patch.set_facecolor('#1a1a1a')
        ax.set_facecolor('#2a2a2a')

        if len(data) > 0:
            ax.plot(list(data), color=color, linewidth=2.5, alpha=0.9)
            ax.fill_between(range(len(data)), list(data), alpha=0.4, color=color)

        ax.set_title(title, color='white', fontsize=11, pad=8, weight='bold')
        ax.set_ylabel(ylabel, color='white', fontsize=9)
        ax.tick_params(colors='white', labelsize=8)
        ax.grid(True, alpha=0.3, color='white', linestyle='--')
        for side in ax.spines.values():
            side.set_color('#666666')

        plt.tight_layout()

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        chart = np.asarray(buf)
        plt.close(fig)

        return cv2.cvtColor(chart, cv2.COLOR_RGBA2BGR)


    def create_gauge(self, value, title, max_value, width, height):
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100, subplot_kw={'projection': 'polar'})
        fig.patch.set_facecolor('#1a1a1a')
        ax.set_facecolor('#1a1a1a')

        theta = np.linspace(0, np.pi, 100)
        ax.plot(theta, [1]*len(theta), color='#3a3a3a', linewidth=22)

        norm = min(value/max_value, 1.0)
        theta_val = theta[:int(len(theta)*norm)]

        if norm < 0.3:
            c = '#00ff00'
        elif norm < 0.7:
            c = '#ffaa00'
        else:
            c = '#ff0000'

        ax.plot(theta_val, [1]*len(theta_val), color=c, linewidth=22)

        ax.text(np.pi/2, 0.5, f"{value:.2f}", ha='center', va='center', fontsize=26, color='white', weight='bold')
        ax.text(np.pi/2, -0.25, title, ha='center', va='center', fontsize=12, color='white', weight='bold')

        ax.set_ylim(0, 1.2)
        ax.axis('off')
        plt.tight_layout()

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        gauge = np.asarray(buf)
        plt.close(fig)

        return cv2.cvtColor(gauge, cv2.COLOR_RGBA2BGR)


    def create_stat_panel(self, width, height):
        p = np.zeros((height, width, 3), dtype=np.uint8)
        p[:] = (26, 26, 26)
        cv2.rectangle(p, (0, 0), (width, 35), (40, 40, 40), -1)
        cv2.putText(p, 'SYSTEM STATISTICS', (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2)

        uptime = time.time() - self.start_time
        avg_fps = np.mean(self.fps_history) if len(self.fps_history)>0 else 0
        risk_pct = (self.high_risk_frames/max(self.total_frames,1))*100

        stats = [
            ("Uptime:", f"{int(uptime//60)}m {int(uptime%60)}s", (100,200,255)),
            ("Frames:", f"{self.total_frames}", (100,255,100)),
            ("Avg FPS:", f"{avg_fps:.1f}", (255,200,100)),
            ("Risk:", f"{risk_pct:.1f}%", (255,100,100)),
        ]

        y = 62
        for label, val, col in stats:
            cv2.putText(p, label, (15,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200),1)
            cv2.putText(p, val, (190,y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col,2)
            y += 28

        return p


    def create_alert_panel(self, width, height):
        p = np.zeros((height,width,3),dtype=np.uint8)
        p[:] = (26,26,26)
        cv2.rectangle(p,(0,0),(width,35),(40,40,40),-1)
        cv2.putText(p,'ACTIVE ALERTS',(15,24),cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,100,100),2)

        y=62
        if not self.current_alerts:
            cv2.putText(p,'All systems normal - No alerts',(15,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(100,255,100),1)
        else:
            for a in self.current_alerts[-3:]:
                c={'high':(0,0,255),'medium':(0,165,255),'low':(0,255,255)}.get(a['severity'],(255,255,255))
                cv2.circle(p,(20,y-6),6,c,-1)
                cv2.putText(p,a['message'][:55],(35,y),cv2.FONT_HERSHEY_SIMPLEX,0.45,c,1)
                y+=28

        return p


    def create_heatmap_legend(self, width, height):
        p = np.zeros((height,width,3),dtype=np.uint8)
        p[:] = (26,26,26)
        cv2.rectangle(p,(0,0),(width,35),(40,40,40),-1)
        cv2.putText(p,'DENSITY SCALE',(15,24),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,100),2)

        gradient = np.zeros((65,240,3),dtype=np.uint8)
        for i in range(240):
            v = int((i/240)*255)
            gradient[:,i] = cv2.applyColorMap(np.array([[v]],dtype=np.uint8),cv2.COLORMAP_JET)[0][0]

        p[48:113,20:260]=gradient
        cv2.putText(p,'Low',(20,130),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),1)
        cv2.putText(p,'Medium',(110,130),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),1)
        cv2.putText(p,'High',(220,130),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),1)

        return p


    def render_dashboard(self, main_frame, model_accuracy=92.5):

        current_density = self.density_history[-1] if self.density_history else 0
        current_risk = self.risk_history[-1] if self.risk_history else 0
        current_motion = self.motion_history[-1] if self.motion_history else 0
        current_fps = self.fps_history[-1] if self.fps_history else 0


        # Cache refresh
        if self.total_frames % 100 == 0 or self.cached_density_chart is None:
            self.cached_density_chart = self.create_line_chart(self.density_history,'Crowd Density Trend','#00ffff','Density',450,220)
            self.cached_risk_chart = self.create_line_chart(self.risk_history,'Risk Level Trend','#ff4444','Risk',450,220)
            self.cached_motion_chart = self.create_line_chart(self.motion_history,'Motion Activity','#44ff44','Motion',450,220)
            self.cached_risk_gauge = self.create_gauge(current_risk,'RISK LEVEL',1.0,280,200)
            self.cached_density_gauge = self.create_gauge(current_density,'DENSITY',1.0,280,200)


        if self.total_frames % 200 == 0 or self.cached_stat_panel is None:
            self.cached_stat_panel = self.create_stat_panel(420,140)
            self.cached_alert_panel = self.create_alert_panel(520,140)
            self.cached_heatmap_legend = self.create_heatmap_legend(280,140)

        density_chart = self.cached_density_chart
        risk_chart = self.cached_risk_chart
        motion_chart = self.cached_motion_chart
        risk_gauge = self.cached_risk_gauge
        density_gauge = self.cached_density_gauge
        stat_panel = self.cached_stat_panel
        alert_panel = self.cached_alert_panel
        heatmap = self.cached_heatmap_legend


        dashboard = np.zeros((1080,1920,3),dtype=np.uint8)
        dashboard[:] = (20,20,20)

        main_w, main_h = 1180, 885
        main_resized = cv2.resize(main_frame,(main_w,main_h))
        dashboard[75:75+main_h,20:20+main_w]=main_resized


        cv2.putText(dashboard,'CROWD SAFETY MONITORING SYSTEM',(35,42),cv2.FONT_HERSHEY_SIMPLEX,1.3,(255,255,255),3)

        cv2.rectangle(dashboard,(1720,15),(1900,55),(40,60,40),-1)
        cv2.putText(dashboard,f"FPS: {current_fps:.1f}",(1730,43),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)

        x=1220
        y=75
        dashboard[y:y+200,x:x+280]=risk_gauge
        dashboard[y:y+200,x+300:x+580]=density_gauge
        y+=215
        dashboard[y:y+220,x:x+450]=density_chart
        y+=230
        dashboard[y:y+220,x:x+450]=risk_chart
        y+=230
        dashboard[y:y+220,x:x+450]=motion_chart

        dashboard[940:1080, 20:440]=stat_panel
        dashboard[940:1080, 460:980]=alert_panel
        dashboard[940:1080, 1000:1280]=heatmap

        status_color = (0,255,0) if current_risk<0.5 else (0,165,255) if current_risk<0.7 else (0,0,255)
        status_text = 'SAFE' if current_risk<0.5 else 'CAUTION' if current_risk<0.7 else 'DANGER'
        cv2.circle(dashboard,(1830,35),15,status_color,-1)
        cv2.putText(dashboard,status_text,(1700,43),cv2.FONT_HERSHEY_SIMPLEX,0.7,status_color,2)

        return dashboard
