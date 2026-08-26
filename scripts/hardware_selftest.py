#!/usr/bin/env python3
import os, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from agrivision.config import load_config
from agrivision.hardware import make_hardware
from agrivision.logic import HealthStatus

cfg=load_config(ROOT/'config.yaml' if (ROOT/'config.yaml').exists() else ROOT/'config.example.yaml')
sim=os.environ.get('AGRIVISION_SIMULATION','0') in {'1','true','yes'}
hw=make_hardware(cfg,sim)
try:
    print('Environment:',hw.read_environment())
    for s in (HealthStatus.HEALTHY,HealthStatus.STRESS,HealthStatus.DISEASE):
        print('LED:',s.value); hw.set_health(s); time.sleep(.7)
    hw.set_health(HealthStatus.NOT_SCANNED)
    print('Buzzer'); hw.beep(.15)
    print('Pump pulse: 1 second'); hw.pump_on(); time.sleep(1); hw.pump_off()
    out=ROOT/'captures'/'selftest.jpg'; print('Camera capture:',out); hw.capture(out)
    print('PASS: hardware self-test completed')
finally:
    hw.close()
