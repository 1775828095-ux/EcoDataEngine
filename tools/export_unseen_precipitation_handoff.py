"""Reuse the actual writer while preserving the frozen study helper unchanged."""
from ecodata.models.adapters.swat_plus import SWATPlusWriter
import evaluate_unseen_portrait_response as frozen

# Serialization belongs to SWATPlusWriter, not SWATPlusAdapter. Only replace
# this immutable helper's local binding; no platform class or prediction changes.
frozen.SWATPlusAdapter=SWATPlusWriter

if __name__=='__main__':frozen.weather()
