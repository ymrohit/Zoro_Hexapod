"""Simple FastAPI server exposing APIs to control the hexapod robot."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time

from servo import Servo
from led import Led
from buzzer import Buzzer
from ultrasonic import Ultrasonic
from adc import ADC
from control import Control

app = FastAPI(title="Hexapod Control API")

servo_controller = Servo()
# lazy instantiate Led to avoid user input prompts if params missing
_led = None
buzzer = Buzzer()
ultrasonic = Ultrasonic()
adc = ADC()
control = Control()

# remember last angles for reporting
servo_state = {i: 90 for i in range(32)}

LEG_MAP = {
    1: [15, 14, 13],
    2: [12, 11, 10],
    3: [9, 8, 31],
    4: [22, 23, 27],
    5: [19, 20, 21],
    6: [16, 17, 18],
}

class Angle(BaseModel):
    angle: int

class LEDCommand(BaseModel):
    mode: str
    r: int = 0
    g: int = 0
    b: int = 0

class BuzzerCommand(BaseModel):
    state: bool

class MoveCommand(BaseModel):
    """Parameters for basic gait movement."""
    gait: int
    x: int = 0
    y: int = 0
    speed: int = 10
    angle: int = 0

class RelaxCommand(BaseModel):
    state: bool

class ServoPowerCommand(BaseModel):
    enabled: bool


class LegAngles(BaseModel):
    coxa: int
    femur: int
    tibia: int

class SweepCommand(BaseModel):
    start: int = 0
    end: int = 180
    step: int = 10
    delay: float = 0.05

@app.get("/parts")
def list_parts():
    led_count = None
    if _led:
        led_count = _led.strip.get_led_count()
    return {
        "servos": list(range(32)),
        "legs": LEG_MAP,
        "servo_range": [0, 180],
        "servo_state": servo_state,
        "led_count": led_count,
        "has_buzzer": True,
        "has_ultrasonic": True,
    }

@app.post("/servo/{channel}")
def set_servo(channel: int, data: Angle):
    if channel < 0 or channel >= 32:
        raise HTTPException(status_code=400, detail="Channel out of range")
    angle = max(0, min(180, data.angle))
    servo_controller.set_servo_angle(channel, angle)
    servo_state[channel] = angle
    return {"channel": channel, "angle": angle}


@app.get("/servo/{channel}")
def get_servo(channel: int):
    if channel < 0 or channel >= 32:
        raise HTTPException(status_code=400, detail="Channel out of range")
    return {"channel": channel, "angle": servo_state.get(channel)}


@app.post("/servo/{channel}/sweep")
def sweep_servo(channel: int, cmd: SweepCommand = SweepCommand()):
    if channel < 0 or channel >= 32:
        raise HTTPException(status_code=400, detail="Channel out of range")
    for a in range(cmd.start, cmd.end + 1, cmd.step):
        servo_controller.set_servo_angle(channel, a)
        servo_state[channel] = a
        time.sleep(cmd.delay)
    return {"channel": channel, "swept": True}


@app.post("/leg/{leg_id}")
def set_leg(leg_id: int, angles: LegAngles):
    if leg_id not in LEG_MAP:
        raise HTTPException(status_code=400, detail="Invalid leg id")
    ch = LEG_MAP[leg_id]
    vals = [angles.coxa, angles.femur, angles.tibia]
    for c, v in zip(ch, vals):
        v = max(0, min(180, v))
        servo_controller.set_servo_angle(c, v)
        servo_state[c] = v
    return {"leg": leg_id, "angles": vals}


@app.get("/leg/{leg_id}")
def get_leg(leg_id: int):
    if leg_id not in LEG_MAP:
        raise HTTPException(status_code=400, detail="Invalid leg id")
    ch = LEG_MAP[leg_id]
    return {"leg": leg_id, "angles": [servo_state[c] for c in ch]}


@app.post("/leg/{leg_id}/sweep")
def sweep_leg(leg_id: int, cmd: SweepCommand = SweepCommand()):
    if leg_id not in LEG_MAP:
        raise HTTPException(status_code=400, detail="Invalid leg id")
    ch = LEG_MAP[leg_id]
    for a in range(cmd.start, cmd.end + 1, cmd.step):
        for c in ch:
            servo_controller.set_servo_angle(c, a)
            servo_state[c] = a
        time.sleep(cmd.delay)
    return {"leg": leg_id, "swept": True}


@app.post("/relax")
def relax_servos(cmd: RelaxCommand):
    """Relax or unrelax all servos."""
    control.relax(cmd.state)
    return {"relaxed": cmd.state}


@app.post("/servo_power")
def servo_power(cmd: ServoPowerCommand):
    """Enable or disable servo power."""
    if cmd.enabled:
        control.servo_power_disable.off()
    else:
        control.servo_power_disable.on()
    return {"enabled": cmd.enabled}


@app.post("/move")
def move_robot(cmd: MoveCommand):
    """Run a basic gait using Control.run_gait."""
    data = ["CMD_MOVE", str(cmd.gait), str(cmd.x), str(cmd.y), str(cmd.speed), str(cmd.angle)]
    control.run_gait(data)
    return {"executed": True}

@app.post("/led")
def led_control(cmd: LEDCommand):
    global _led
    if _led is None:
        _led = Led()
    data = [cmd.mode, str(cmd.r), str(cmd.g), str(cmd.b)]
    _led.process_light_command(data)
    return {"status": "ok"}

@app.post("/buzzer")
def buzzer_control(cmd: BuzzerCommand):
    buzzer.set_state(cmd.state)
    return {"state": cmd.state}

@app.get("/distance")
def get_distance():
    distance = ultrasonic.get_distance()
    return {"distance_cm": distance}

@app.get("/battery")
def get_battery():
    voltage = adc.read_battery_voltage()
    return {"v1": voltage[0], "v2": voltage[1]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
