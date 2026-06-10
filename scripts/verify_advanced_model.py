import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../Backend'))
from utils.model_utils import AdvancedCNNLSTMDetector
from app import HeuristicDecisionFlow

def test_shape_flow():
    print("====================================================")
    print("Testing AdvancedCNNLSTMDetector Shape Flow...")
    print("====================================================")
    
    # Batch size = 2, Sequence Length = 16, 3 channels, 299x299
    dummy_input = torch.randn(2, 16, 3, 299, 299)
    print(f"Input shape: {dummy_input.shape}")
    
    # Instantiate model
    model = AdvancedCNNLSTMDetector(use_pretrained=False)
    model.eval()
    
    # Run forward pass
    with torch.no_grad():
        logits, consistency_score, spike_flags, deltas = model(dummy_input)
        
    print(f"Logits shape           : {logits.shape}")
    print(f"Consistency score shape: {consistency_score.shape}")
    print(f"Spike flags shape      : {spike_flags.shape}")
    print(f"Deltas shape           : {deltas.shape}")
    
    assert logits.shape == (2, 2), f"Expected logits shape (2, 2), got {logits.shape}"
    assert consistency_score.shape == (32, 1), f"Expected consistency shape (32, 1), got {consistency_score.shape}"
    assert spike_flags.shape == (2, 15), f"Expected spike flags shape (2, 15), got {spike_flags.shape}"
    assert deltas.shape == (2, 15), f"Expected deltas shape (2, 15), got {deltas.shape}"
    
    print("[OK] Model shape flow is completely correct!")
    print()

def test_heuristic_rules():
    print("====================================================")
    print("Testing Heuristic Decision Flow Rules (H-1 to H-7)...")
    print("====================================================")
    
    # Test H-7: Confidence Recovery
    # raw_score=0.95 (highly confident fake), no anomalies -> FAKE
    res = HeuristicDecisionFlow.process(raw_score=0.95, rppg_consistency=0.9, is_video=True)
    print(f"H-7 Test: raw_score=0.95, rppg=0.9 -> verdict={res['verdict']}, confidence={res['confidence']:.2f}%, rules={res['triggered_rules']}")
    assert res['verdict'] == 'Fake'
    assert 'H-7' in res['triggered_rules']
    
    # Test H-1: Low-Confidence Gate with 0 flags -> Real
    res = HeuristicDecisionFlow.process(raw_score=0.60, rppg_consistency=0.8, is_video=True)
    print(f"H-1 Test (0 flags): raw_score=0.60, rppg=0.8 -> verdict={res['verdict']}, confidence={res['confidence']:.2f}%, rules={res['triggered_rules']}")
    assert res['verdict'] == 'Real'
    assert 'H-1' in res['triggered_rules']

    # Test H-1: Low-Confidence Gate with exactly 1 flag (H-4) -> Uncertain
    res = HeuristicDecisionFlow.process(raw_score=0.60, rppg_consistency=0.8, spectral_aliasing=0.90, is_video=True)
    print(f"H-1 Test (1 flag): raw_score=0.60, rppg=0.8, spectral=0.9 -> verdict={res['verdict']}, confidence={res['confidence']:.2f}%, rules={res['triggered_rules']}")
    assert res['verdict'] == 'Uncertain'
    assert 'H-1' in res['triggered_rules']
    assert 'H-4' in res['triggered_rules']

    # Test H-2: Bio-Consistency Hard Veto
    # raw_score=0.10 (model says Real), rppg_consistency=0.15 (anomalous heart-rate correlation) -> Override to FAKE
    res = HeuristicDecisionFlow.process(raw_score=0.10, rppg_consistency=0.15, is_video=True)
    print(f"H-2 Test: raw_score=0.10, rppg=0.15 -> verdict={res['verdict']}, confidence={res['confidence']:.2f}%, rules={res['triggered_rules']}")
    assert res['verdict'] == 'Fake'
    assert 'H-2' in res['triggered_rules']
    assert res['is_override'] is True

    # Test H-3: Spike Density Rule
    # raw_score=0.40, 6 of 16 frames are anomalies -> Override to FAKE
    spikes = [True]*6 + [False]*10
    res = HeuristicDecisionFlow.process(raw_score=0.40, rppg_consistency=0.8, spike_flags=spikes, is_video=True)
    print(f"H-3 Test: raw_score=0.40, spike_count=6 -> verdict={res['verdict']}, confidence={res['confidence']:.2f}%, rules={res['triggered_rules']}")
    assert res['verdict'] == 'Fake'
    assert 'H-3' in res['triggered_rules']
    assert res['is_override'] is True

    # Test H-4 & H-5: Soft boosts
    # raw_score=0.60, spectral_aliasing=0.90 (+0.15 boost), rppg_variance=4.0 (+0.10 boost) -> raw 0.60 + 0.25 = 0.85 Fake confidence
    res = HeuristicDecisionFlow.process(raw_score=0.60, rppg_consistency=0.8, spectral_aliasing=0.90, rppg_variance=4.0, is_video=True)
    print(f"H-4 & H-5 Test: raw_score=0.60, spectral_aliasing=0.90, rppg_var=4.0 -> verdict={res['verdict']}, confidence={res['confidence']:.2f}%, rules={res['triggered_rules']}")
    assert res['verdict'] == 'Fake'
    assert abs(res['adjusted_score'] - 0.85) < 1e-4
    assert 'H-4' in res['triggered_rules']
    assert 'H-5' in res['triggered_rules']

    # Test H-6: Temporal Freeze Rule
    # raw_score=0.30, frame deltas has 5 zeros -> Override to FAKE
    deltas = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    res = HeuristicDecisionFlow.process(raw_score=0.30, rppg_consistency=0.8, frame_deltas=deltas, is_video=True)
    print(f"H-6 Test: raw_score=0.30, consecutive zeros -> verdict={res['verdict']}, confidence={res['confidence']:.2f}%, rules={res['triggered_rules']}")
    assert res['verdict'] == 'Fake'
    assert 'H-6' in res['triggered_rules']
    assert res['is_override'] is True

    print("[OK] Heuristic decision flow rules pass verification!")
    print()

if __name__ == '__main__':
    test_shape_flow()
    test_heuristic_rules()
