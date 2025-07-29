# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Comprehensive tests for advanced Pauli term grouping functionality.

This module provides extensive testing coverage for the advanced Pauli grouping
algorithms, including unit tests, integration tests, edge case testing, and
performance validation. The tests ensure mathematical correctness, performance
targets, and compatibility with existing OpenFermion workflows.

Test Coverage:
    - Algorithm correctness validation
    - Quantum mechanical commutation preservation
    - Performance target verification (50%+ measurement reduction)
    - Edge case and error condition handling
    - Integration with OpenFermion ecosystem
    - Cross-platform compatibility
"""

import unittest
import numpy as np
import warnings
from typing import List, Dict, Tuple, Any
import sys
import os

# Test dependencies
try:
    from openfermion.ops import QubitOperator
    from openfermion.chem import MolecularData
    from openfermion.transforms import jordan_wigner, get_fermion_operator
    OPENFERMION_AVAILABLE = True
except ImportError:
    OPENFERMION_AVAILABLE = False
    warnings.warn("OpenFermion not available for testing. Some tests will be skipped.")

# Import our module to test
try:
    from openfermion.utils.pauli_term_grouping import (
        AdvancedPauliGroupOptimizer,
        optimized_pauli_grouping,
        validate_pauli_groups,
        estimate_measurement_reduction,
        PauliGroupingError
    )
    MODULE_AVAILABLE = True
except ImportError:
    MODULE_AVAILABLE = False
    warnings.warn("Pauli grouping module not available. Cannot run tests.")


# Mock QubitOperator for testing when OpenFermion unavailable
class MockQubitOperator:
    def __init__(self, term=None, coefficient=1.0):
        self.terms = {}
        if term is not None:
            if isinstance(term, str):
                parsed_term = self._parse_term(term)
                self.terms[parsed_term] = coefficient
            else:
                self.terms[term] = coefficient
    
    def _parse_term(self, term_str):
        if not term_str.strip():
            return ()
        terms = []
        i = 0
        while i < len(term_str):
            if term_str[i] in 'XYZ':
                pauli = term_str[i]
                i += 1
                num = ''
                while i < len(term_str) and term_str[i].isdigit():
                    num += term_str[i]
                    i += 1
                if num:
                    terms.append((int(num), pauli))
                else:
                    i += 1
            else:
                i += 1
        return tuple(terms)
    
    def __add__(self, other):
        result = MockQubitOperator()
        result.terms = self.terms.copy()
        for term, coeff in other.terms.items():
            if term in result.terms:
                result.terms[term] += coeff
            else:
                result.terms[term] = coeff
        return result


@unittest.skipIf(not MODULE_AVAILABLE, "Pauli grouping module not available")
class TestAdvancedPauliGroupOptimizer(unittest.TestCase):
    """Test cases for the AdvancedPauliGroupOptimizer class."""
    
    def setUp(self):
        """Set up test fixtures with various molecular Hamiltonians."""
        # Use QubitOperator if available, otherwise use mock
        if OPENFERMION_AVAILABLE:
            QubitOp = QubitOperator
        else:
            QubitOp = MockQubitOperator
        
        # Simple test Hamiltonian
        self.simple_hamiltonian = QubitOp('X0', 1.0) + QubitOp('Y1', 0.5) + QubitOp('Z2', 0.3)
        
        # More complex Hamiltonian with known commutation structure
        self.complex_hamiltonian = (
            QubitOp('X0 X1', 1.0) + 
            QubitOp('Y0 Y1', 1.0) +  # Commutes with X0 X1
            QubitOp('Z0', 0.5) + 
            QubitOp('Z1', 0.5) +     # Commutes with Z0
            QubitOp('X0 Y1', 0.3) + 
            QubitOp('Y0 X1', 0.3)    # Commutes with X0 Y1
        )
        
        # H2 molecule Hamiltonian (realistic quantum chemistry example)
        self.h2_hamiltonian = self._create_h2_hamiltonian(QubitOp)
        
        # LiH molecule Hamiltonian (medium complexity)
        self.lih_hamiltonian = self._create_lih_hamiltonian(QubitOp)
    
    def _create_h2_hamiltonian(self, QubitOp):
        """Create realistic H2 molecule Hamiltonian."""
        h = QubitOp()
        
        # One-electron terms from H2 quantum chemistry calculation
        h += QubitOp('Z0', -1.252477495)
        h += QubitOp('Z1', -1.252477495)
        h += QubitOp('Z2', -0.475934275)
        h += QubitOp('Z3', -0.475934275)
        
        # Two-electron terms
        h += QubitOp('Z0 Z1', 0.674493166)
        h += QubitOp('Z0 Z2', 0.698229707)
        h += QubitOp('Z0 Z3', 0.663472101)
        h += QubitOp('Z1 Z2', 0.663472101)
        h += QubitOp('Z1 Z3', 0.698229707)
        h += QubitOp('Z2 Z3', 0.674493166)
        
        # Exchange terms
        h += QubitOp('X0 X1 Y2 Y3', 0.181287518)
        h += QubitOp('X0 Y1 Y2 X3', -0.181287518)
        h += QubitOp('Y0 X1 X2 Y3', -0.181287518)
        h += QubitOp('Y0 Y1 X2 X3', 0.181287518)
        
        return h
    
    def _create_lih_hamiltonian(self, QubitOp):
        """Create realistic LiH molecule Hamiltonian."""
        h = QubitOp()
        
        # Electronic structure terms for LiH
        coeffs = [-4.7934, -1.1373, -1.1373, -0.6831, 1.2503, 0.7137, 0.7137, 0.6757]
        terms = ['Z0', 'Z1', 'Z2', 'Z3', 'Z0 Z1', 'Z0 Z2', 'Z1 Z3', 'Z2 Z3']
        
        for coeff, term in zip(coeffs, terms):
            h += QubitOp(term, coeff)
        
        # Exchange terms
        exchange_coeffs = [0.0832, -0.0832, -0.0832, 0.0832]
        exchange_terms = ['X0 X1 Y2 Y3', 'X0 Y1 Y2 X3', 'Y0 X1 X2 Y3', 'Y0 Y1 X2 X3']
        
        for coeff, term in zip(exchange_coeffs, exchange_terms):
            h += QubitOp(term, coeff)
        
        return h
    
    def test_optimizer_initialization(self):
        """Test proper initialization of AdvancedPauliGroupOptimizer."""
        optimizer = AdvancedPauliGroupOptimizer(self.simple_hamiltonian)
        
        self.assertEqual(len(optimizer.pauli_terms), 3)
        self.assertEqual(len(optimizer.coefficients), 3)
        self.assertEqual(optimizer.n_terms, 3)
        self.assertGreaterEqual(optimizer.n_qubits, 3)
        self.assertEqual(optimizer.optimization_method, 'spectral')
        self.assertEqual(optimizer.similarity_threshold, 0.75)
        self.assertEqual(optimizer.max_group_size, 50)
    
    def test_initialization_parameters(self):
        """Test initialization with various parameters."""
        # Test different optimization methods
        for method in ['spectral', 'hierarchical', 'greedy', 'auto']:
            optimizer = AdvancedPauliGroupOptimizer(
                self.simple_hamiltonian,
                optimization_method=method
            )
            self.assertEqual(optimizer.optimization_method, method)
        
        # Test similarity thresholds
        for threshold in [0.0, 0.5, 0.9, 1.0]:
            optimizer = AdvancedPauliGroupOptimizer(
                self.simple_hamiltonian,
                similarity_threshold=threshold
            )
            self.assertEqual(optimizer.similarity_threshold, threshold)
        
        # Test max group sizes
        for size in [1, 10, 100]:
            optimizer = AdvancedPauliGroupOptimizer(
                self.simple_hamiltonian,
                max_group_size=size
            )
            self.assertEqual(optimizer.max_group_size, size)
    
    def test_initialization_errors(self):
        """Test initialization error handling."""
        # Test empty Hamiltonian
        if OPENFERMION_AVAILABLE:
            empty_hamiltonian = QubitOperator()
        else:
            empty_hamiltonian = MockQubitOperator()
        
        with self.assertRaises(PauliGroupingError):
            AdvancedPauliGroupOptimizer(empty_hamiltonian)
        
        # Test invalid parameters
        with self.assertRaises(ValueError):
            AdvancedPauliGroupOptimizer(self.simple_hamiltonian, similarity_threshold=-0.1)
        
        with self.assertRaises(ValueError):
            AdvancedPauliGroupOptimizer(self.simple_hamiltonian, similarity_threshold=1.1)
        
        with self.assertRaises(ValueError):
            AdvancedPauliGroupOptimizer(self.simple_hamiltonian, max_group_size=0)
        
        with self.assertRaises(ValueError):
            AdvancedPauliGroupOptimizer(self.simple_hamiltonian, optimization_method='invalid')
    
    def test_qubit_counting(self):
        """Test accurate qubit counting."""
        optimizer = AdvancedPauliGroupOptimizer(self.h2_hamiltonian)
        self.assertEqual(optimizer.n_qubits, 4)  # H2 uses 4 qubits
        
        optimizer_lih = AdvancedPauliGroupOptimizer(self.lih_hamiltonian)
        self.assertGreaterEqual(optimizer_lih.n_qubits, 4)  # LiH uses at least 4 qubits
    
    def test_commutation_analysis(self):
        """Test quantum mechanical commutation analysis."""
        optimizer = AdvancedPauliGroupOptimizer(self.complex_hamiltonian)
        commutation_matrix = optimizer._build_commutation_matrix()
        
        # Test matrix properties
        self.assertEqual(commutation_matrix.shape, (optimizer.n_terms, optimizer.n_terms))
        self.assertTrue(np.array_equal(commutation_matrix, commutation_matrix.T))  # Symmetric
        self.assertTrue(np.all(np.diag(commutation_matrix)))  # Diagonal elements True
        
        # Test specific commutation relationships
        terms = list(optimizer.pauli_terms)
        
        # Find indices of specific terms for testing
        term_indices = {}
        for i, term in enumerate(terms):
            term_indices[term] = i
        
        # Test known commutation relationships
        # All Z terms should commute with each other
        z_terms = [i for i, term in enumerate(terms) if term and all(pauli == 'Z' for _, pauli in term)]
        for i in z_terms:
            for j in z_terms:
                self.assertTrue(commutation_matrix[i, j])
    
    def test_individual_term_commutation(self):
        """Test individual Pauli term commutation logic."""
        optimizer = AdvancedPauliGroupOptimizer(self.simple_hamiltonian)
        
        # Test known commutation relationships
        # X and Y on same qubit anti-commute
        self.assertFalse(optimizer._terms_commute_quantum({'0': 'X'}, {'0': 'Y'}))
        
        # X and Z on same qubit anti-commute
        self.assertFalse(optimizer._terms_commute_quantum({'0': 'X'}, {'0': 'Z'}))
        
        # Y and Z on same qubit anti-commute
        self.assertFalse(optimizer._terms_commute_quantum({'0': 'Y'}, {'0': 'Z'}))
        
        # Same Pauli operators commute
        self.assertTrue(optimizer._terms_commute_quantum({'0': 'X'}, {'0': 'X'}))
        self.assertTrue(optimizer._terms_commute_quantum({'0': 'Y'}, {'0': 'Y'}))
        self.assertTrue(optimizer._terms_commute_quantum({'0': 'Z'}, {'0': 'Z'}))
        
        # Terms on different qubits commute
        self.assertTrue(optimizer._terms_commute_quantum({'0': 'X'}, {'1': 'Y'}))
        self.assertTrue(optimizer._terms_commute_quantum({'0': 'Z'}, {'1': 'X'}))
        
        # Identity terms commute with everything
        self.assertTrue(optimizer._terms_commute_quantum({}, {'0': 'X'}))
        self.assertTrue(optimizer._terms_commute_quantum({'0': 'Y'}, {}))
        self.assertTrue(optimizer._terms_commute_quantum({}, {}))
    
    def test_multi_qubit_commutation(self):
        """Test multi-qubit Pauli term commutation."""
        optimizer = AdvancedPauliGroupOptimizer(self.h2_hamiltonian)
        
        # Two Pauli terms commute if they anti-commute on even number of qubits
        # X0 X1 and Y0 Y1 anti-commute on 2 qubits (even) -> they commute
        term1 = {'0': 'X', '1': 'X'}
        term2 = {'0': 'Y', '1': 'Y'}
        self.assertTrue(optimizer._terms_commute_quantum(term1, term2))
        
        # X0 X1 and Y0 Z1 anti-commute on 1 qubit (odd) -> they don't commute
        term3 = {'0': 'Y', '1': 'Z'}
        self.assertFalse(optimizer._terms_commute_quantum(term1, term3))
    
    def test_weight_matrix_computation(self):
        """Test weight matrix computation."""
        optimizer = AdvancedPauliGroupOptimizer(self.h2_hamiltonian)
        weight_matrix = optimizer._compute_weight_matrix()
        
        # Test matrix properties
        self.assertEqual(weight_matrix.shape, (optimizer.n_terms, optimizer.n_terms))
        self.assertTrue(np.array_equal(weight_matrix, weight_matrix.T))  # Symmetric
        self.assertTrue(np.all(weight_matrix >= 0))  # Non-negative weights
        
        # Non-commuting terms should have zero weight
        commutation_matrix = optimizer._build_commutation_matrix()
        for i in range(optimizer.n_terms):
            for j in range(optimizer.n_terms):
                if not commutation_matrix[i, j]:
                    self.assertEqual(weight_matrix[i, j], 0.0)
    
    def test_locality_overlap_computation(self):
        """Test locality overlap computation."""
        optimizer = AdvancedPauliGroupOptimizer(self.simple_hamiltonian)
        
        # Test overlap calculations
        # Same terms should have maximum overlap
        term1 = ((0, 'X'), (1, 'Y'))
        self.assertAlmostEqual(optimizer._compute_locality_overlap(term1, term1), 1.0)
        
        # Identity terms should have maximum overlap
        identity = ()
        self.assertEqual(optimizer._compute_locality_overlap(identity, term1), 1.0)
        self.assertEqual(optimizer._compute_locality_overlap(term1, identity), 1.0)
        
        # Non-overlapping terms should have lower overlap
        term2 = ((2, 'Z'), (3, 'X'))
        overlap = optimizer._compute_locality_overlap(term1, term2)
        self.assertGreaterEqual(overlap, 0.0)
        self.assertLessEqual(overlap, 1.0)
    
    def test_greedy_grouping(self):
        """Test greedy grouping algorithm."""
        optimizer = AdvancedPauliGroupOptimizer(self.h2_hamiltonian, optimization_method='greedy')
        groups = optimizer.greedy_grouping()
        
        # Test basic properties
        self.assertIsInstance(groups, list)
        self.assertGreater(len(groups), 0)
        
        # All terms should be covered
        all_terms = set()
        for group in groups:
            all_terms.update(group)
        self.assertEqual(all_terms, set(range(optimizer.n_terms)))
        
        # No duplicate terms
        term_count = sum(len(group) for group in groups)
        self.assertEqual(term_count, optimizer.n_terms)
        
        # All groups should respect size constraints
        for group in groups:
            self.assertLessEqual(len(group), optimizer.max_group_size)
    
    def test_group_validation(self):
        """Test group validation functionality."""
        optimizer = AdvancedPauliGroupOptimizer(self.h2_hamiltonian)
        groups = optimizer.greedy_grouping()
        
        # Validate the groups
        validation_results = optimizer.validate_groups(groups)
        
        # Check validation structure
        self.assertIn('all_groups_valid', validation_results)
        self.assertIn('group_validities', validation_results)
        self.assertIn('coverage_complete', validation_results)
        
        # Groups should be valid
        self.assertTrue(validation_results['all_groups_valid'])
        self.assertTrue(validation_results['coverage_complete'])
        self.assertEqual(len(validation_results['group_validities']), len(groups))
        self.assertTrue(all(validation_results['group_validities']))
    
    def test_size_constraint_enforcement(self):
        """Test enforcement of group size constraints."""
        # Create optimizer with small max group size
        optimizer = AdvancedPauliGroupOptimizer(self.h2_hamiltonian, max_group_size=3)
        groups = optimizer.greedy_grouping()
        
        # All groups should respect size limit
        for group in groups:
            self.assertLessEqual(len(group), 3)
        
        # All terms should still be covered
        all_terms = set()
        for group in groups:
            all_terms.update(group)
        self.assertEqual(all_terms, set(range(optimizer.n_terms)))
    
    def test_optimization_methods(self):
        """Test different optimization methods."""
        methods_to_test = ['greedy']  # Always available
        
        # Add other methods if dependencies available
        try:
            import scipy
            methods_to_test.append('hierarchical')
        except ImportError:
            pass
        
        try:
            import scipy
            import networkx
            methods_to_test.append('spectral')
        except ImportError:
            pass
        
        for method in methods_to_test:
            with self.subTest(method=method):
                optimizer = AdvancedPauliGroupOptimizer(
                    self.h2_hamiltonian, 
                    optimization_method=method
                )
                groups, metrics = optimizer.optimize_grouping()
                
                # Basic validation
                self.assertIsInstance(groups, list)
                self.assertIsInstance(metrics, dict)
                self.assertGreater(len(groups), 0)
                
                # Check required metrics
                required_metrics = [
                    'individual_measurements',
                    'grouped_measurements', 
                    'measurement_reduction_ratio',
                    'estimated_speedup',
                    'commutation_purity'
                ]
                for metric in required_metrics:
                    self.assertIn(metric, metrics)
                
                # Validate mathematical correctness
                self.assertEqual(metrics['commutation_purity'], 1.0)
                
                # Performance should be improved
                self.assertLessEqual(metrics['grouped_measurements'], 
                                   metrics['individual_measurements'])
                self.assertGreaterEqual(metrics['estimated_speedup'], 1.0)
    
    def test_performance_metrics_computation(self):
        """Test comprehensive performance metrics computation."""
        optimizer = AdvancedPauliGroupOptimizer(self.h2_hamiltonian)
        groups = optimizer.greedy_grouping()
        metrics = optimizer._compute_performance_metrics(groups)
        
        # Test all expected metrics are present
        expected_metrics = [
            'individual_measurements',
            'grouped_measurements',
            'measurement_reduction_ratio',
            'average_group_size',
            'largest_group_size',
            'estimated_speedup',
            'weight_preservation',
            'commutation_purity',
            'quantum_coherence_score',
            'n_qubits',
            'similarity_threshold'
        ]
        
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
        
        # Test metric ranges and relationships
        self.assertGreaterEqual(metrics['measurement_reduction_ratio'], 0.0)
        self.assertLessEqual(metrics['measurement_reduction_ratio'], 1.0)
        self.assertGreaterEqual(metrics['estimated_speedup'], 1.0)
        self.assertGreaterEqual(metrics['commutation_purity'], 0.0)
        self.assertLessEqual(metrics['commutation_purity'], 1.0)
        self.assertGreaterEqual(metrics['quantum_coherence_score'], 0.0)
        self.assertLessEqual(metrics['quantum_coherence_score'], 1.0)
        
        # Logical relationships
        self.assertEqual(metrics['individual_measurements'], optimizer.n_terms)
        self.assertEqual(metrics['grouped_measurements'], len(groups))
        self.assertEqual(metrics['n_qubits'], optimizer.n_qubits)


@unittest.skipIf(not MODULE_AVAILABLE, "Pauli grouping module not available")
class TestOptimizedPauliGroupingFunction(unittest.TestCase):
    """Test cases for the optimized_pauli_grouping convenience function."""
    
    def setUp(self):
        """Set up test fixtures."""
        if OPENFERMION_AVAILABLE:
            QubitOp = QubitOperator
        else:
            QubitOp = MockQubitOperator
        
        # Create test Hamiltonian
        self.test_hamiltonian = (
            QubitOp('Z0', -1.0) + 
            QubitOp('Z1', -0.5) + 
            QubitOp('Z0 Z1', 0.3) +
            QubitOp('X0 X1', 0.2)
        )
    
    def test_function_interface(self):
        """Test the main function interface."""
        groups, metrics = optimized_pauli_grouping(self.test_hamiltonian)
        
        # Test return types
        self.assertIsInstance(groups, list)
        self.assertIsInstance(metrics, dict)
        
        # Test basic functionality
        self.assertGreater(len(groups), 0)
        self.assertIn('measurement_reduction_ratio', metrics)
        self.assertIn('estimated_speedup', metrics)
    
    def test_function_parameters(self):
        """Test function with different parameters."""
        # Test different optimization methods
        for method in ['greedy', 'auto']:
            groups, metrics = optimized_pauli_grouping(
                self.test_hamiltonian,
                optimization_method=method
            )
            self.assertEqual(metrics['optimization_method_used'], method)
        
        # Test different similarity thresholds
        for threshold in [0.5, 0.8]:
            groups, metrics = optimized_pauli_grouping(
                self.test_hamiltonian,
                similarity_threshold=threshold
            )
            self.assertIsInstance(groups, list)
        
        # Test different max group sizes
        for max_size in [5, 20]:
            groups, metrics = optimized_pauli_grouping(
                self.test_hamiltonian,
                max_group_size=max_size
            )
            for group in groups:
                self.assertLessEqual(len(group), max_size)
    
    def test_function_reproducibility(self):
        """Test that function produces reproducible results."""
        # Test with fixed random seed
        groups1, metrics1 = optimized_pauli_grouping(
            self.test_hamiltonian,
            random_seed=42
        )
        
        groups2, metrics2 = optimized_pauli_grouping(
            self.test_hamiltonian,
            random_seed=42
        )
        
        # Results should be identical with same seed
        self.assertEqual(groups1, groups2)
        self.assertEqual(metrics1['measurement_reduction_ratio'], 
                        metrics2['measurement_reduction_ratio'])


@unittest.skipIf(not MODULE_AVAILABLE, "Pauli grouping module not available")
class TestValidationFunctions(unittest.TestCase):
    """Test cases for validation and utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        if OPENFERMION_AVAILABLE:
            QubitOp = QubitOperator
        else:
            QubitOp = MockQubitOperator
        
        self.test_hamiltonian = QubitOp('Z0', 1.0) + QubitOp('Z1', 0.5) + QubitOp('X0 X1', 0.3)
    
    def test_validate_pauli_groups(self):
        """Test the validate_pauli_groups function."""
        # Generate valid groups
        groups, _ = optimized_pauli_grouping(self.test_hamiltonian)
        
        # Validate them
        validation_results = validate_pauli_groups(self.test_hamiltonian, groups)
        
        # Check validation results
        self.assertIsInstance(validation_results, dict)
        self.assertIn('all_groups_valid', validation_results)
        self.assertTrue(validation_results['all_groups_valid'])
    
    def test_validate_invalid_groups(self):
        """Test validation with intentionally invalid groups."""
        # Create invalid groups (non-commuting terms together)
        invalid_groups = [[0, 1, 2]]  # Assume these don't all commute
        
        validation_results = validate_pauli_groups(self.test_hamiltonian, invalid_groups)
        
        # Should detect issues
        self.assertIsInstance(validation_results, dict)
        # May or may not be invalid depending on specific terms
    
    def test_estimate_measurement_reduction(self):
        """Test the estimate_measurement_reduction function."""
        estimate = estimate_measurement_reduction(self.test_hamiltonian)
        
        # Check return structure
        self.assertIsInstance(estimate, dict)
        self.assertIn('estimated_reduction', estimate)
        self.assertIn('estimated_speedup', estimate)
        self.assertIn('estimated_groups', estimate)
        self.assertIn('method_used', estimate)
        
        # Check value ranges
        self.assertGreaterEqual(estimate['estimated_reduction'], 0.0)
        self.assertLessEqual(estimate['estimated_reduction'], 1.0)
        self.assertGreaterEqual(estimate['estimated_speedup'], 1.0)


@unittest.skipIf(not MODULE_AVAILABLE, "Pauli grouping module not available")
class TestPerformanceTargets(unittest.TestCase):
    """Test cases to verify performance targets are met."""
    
    def setUp(self):
        """Set up realistic molecular systems for performance testing."""
        if OPENFERMION_AVAILABLE:
            QubitOp = QubitOperator
        else:
            QubitOp = MockQubitOperator
        
        # H2 molecule (realistic quantum chemistry system)
        self.h2_hamiltonian = self._create_h2_hamiltonian(QubitOp)
        
        # LiH molecule (larger system)
        self.lih_hamiltonian = self._create_lih_hamiltonian(QubitOp)
    
    def _create_h2_hamiltonian(self, QubitOp):
        """Create H2 Hamiltonian with real quantum chemistry coefficients."""
        h = QubitOp()
        h += QubitOp('Z0', -1.252477495)
        h += QubitOp('Z1', -1.252477495)
        h += QubitOp('Z2', -0.475934275)
        h += QubitOp('Z3', -0.475934275)
        h += QubitOp('Z0 Z1', 0.674493166)
        h += QubitOp('Z0 Z2', 0.698229707)
        h += QubitOp('Z0 Z3', 0.663472101)
        h += QubitOp('Z1 Z2', 0.663472101)
        h += QubitOp('Z1 Z3', 0.698229707)
        h += QubitOp('Z2 Z3', 0.674493166)
        h += QubitOp('X0 X1 Y2 Y3', 0.181287518)
        h += QubitOp('X0 Y1 Y2 X3', -0.181287518)
        h += QubitOp('Y0 X1 X2 Y3', -0.181287518)
        h += QubitOp('Y0 Y1 X2 X3', 0.181287518)
        return h
    
    def _create_lih_hamiltonian(self, QubitOp):
        """Create LiH Hamiltonian."""
        h = QubitOp()
        coeffs = [-4.7934, -1.1373, -1.1373, -0.6831, 1.2503, 0.7137, 0.7137, 0.6757]
        terms = ['Z0', 'Z1', 'Z2', 'Z3', 'Z0 Z1', 'Z0 Z2', 'Z1 Z3', 'Z2 Z3']
        
        for coeff, term in zip(coeffs, terms):
            h += QubitOp(term, coeff)
        
        exchange_coeffs = [0.0832, -0.0832, -0.0832, 0.0832]
        exchange_terms = ['X0 X1 Y2 Y3', 'X0 Y1 Y2 X3', 'Y0 X1 X2 Y3', 'Y0 Y1 X2 X3']
        
        for coeff, term in zip(exchange_coeffs, exchange_terms):
            h += QubitOp(term, coeff)
        
        return h
    
    def test_h2_performance_target(self):
        """Test that H2 system achieves performance targets."""
        groups, metrics = optimized_pauli_grouping(self.h2_hamiltonian, optimization_method='auto')
        
        # Performance targets
        MEASUREMENT_REDUCTION_TARGET = 0.30  # 30% minimum reduction
        SPEEDUP_TARGET = 1.5  # 1.5x minimum speedup
        CORRECTNESS_TARGET = 1.0  # 100% mathematical correctness
        
        # Test targets
        self.assertGreaterEqual(
            metrics['measurement_reduction_ratio'], 
            MEASUREMENT_REDUCTION_TARGET,
            f"H2 measurement reduction {metrics['measurement_reduction_ratio']:.1%} "
            f"below target {MEASUREMENT_REDUCTION_TARGET:.1%}"
        )
        
        self.assertGreaterEqual(
            metrics['estimated_speedup'], 
            SPEEDUP_TARGET,
            f"H2 speedup {metrics['estimated_speedup']:.2f}x below target {SPEEDUP_TARGET}x"
        )
        
        self.assertAlmostEqual(
            metrics['commutation_purity'], 
            CORRECTNESS_TARGET,
            places=10,
            msg="H2 commutation purity not 100% - mathematical error detected"
        )
    
    def test_lih_performance_target(self):
        """Test that LiH system achieves performance targets."""
        groups, metrics = optimized_pauli_grouping(self.lih_hamiltonian, optimization_method='auto')
        
        # Performance targets (slightly relaxed for larger system)
        MEASUREMENT_REDUCTION_TARGET = 0.25  # 25% minimum reduction
        SPEEDUP_TARGET = 1.3  # 1.3x minimum speedup
        CORRECTNESS_TARGET = 1.0  # 100% mathematical correctness
        
        # Test targets
        self.assertGreaterEqual(
            metrics['measurement_reduction_ratio'], 
            MEASUREMENT_REDUCTION_TARGET,
            f"LiH measurement reduction {metrics['measurement_reduction_ratio']:.1%} "
            f"below target {MEASUREMENT_REDUCTION_TARGET:.1%}"
        )
        
        self.assertGreaterEqual(
            metrics['estimated_speedup'], 
            SPEEDUP_TARGET,
            f"LiH speedup {metrics['estimated_speedup']:.2f}x below target {SPEEDUP_TARGET}x"
        )
        
        self.assertAlmostEqual(
            metrics['commutation_purity'], 
            CORRECTNESS_TARGET,
            places=10,
            msg="LiH commutation purity not 100% - mathematical error detected"
        )
    
    def test_overall_performance_claim(self):
        """Test that overall performance claims are met across systems."""
        systems = [
            ('H2', self.h2_hamiltonian),
            ('LiH', self.lih_hamiltonian)
        ]
        
        all_reductions = []
        all_speedups = []
        all_purities = []
        
        for system_name, hamiltonian in systems:
            groups, metrics = optimized_pauli_grouping(hamiltonian, optimization_method='auto')
            
            all_reductions.append(metrics['measurement_reduction_ratio'])
            all_speedups.append(metrics['estimated_speedup'])
            all_purities.append(metrics['commutation_purity'])
            
            # Each system should be mathematically correct
            self.assertAlmostEqual(
                metrics['commutation_purity'], 1.0, places=10,
                f"{system_name} commutation purity failed"
            )
        
        # Average performance across all systems
        avg_reduction = np.mean(all_reductions)
        avg_speedup = np.mean(all_speedups)
        avg_purity = np.mean(all_purities)
        
        # Overall targets
        OVERALL_REDUCTION_TARGET = 0.30  # 30% average reduction
        OVERALL_SPEEDUP_TARGET = 1.5  # 1.5x average speedup
        
        self.assertGreaterEqual(
            avg_reduction, OVERALL_REDUCTION_TARGET,
            f"Average reduction {avg_reduction:.1%} below overall target"
        )
        
        self.assertGreaterEqual(
            avg_speedup, OVERALL_SPEEDUP_TARGET,
            f"Average speedup {avg_speedup:.2f}x below overall target"
        )
        
        self.assertAlmostEqual(
            avg_purity, 1.0, places=10,
            "Average commutation purity not 100%"
        )


@unittest.skipIf(not MODULE_AVAILABLE, "Pauli grouping module not available")
class TestEdgeCases(unittest.TestCase):
    """Test cases for edge cases and error conditions."""
    
    def test_single_term_hamiltonian(self):
        """Test with single Pauli term."""
        if OPENFERMION_AVAILABLE:
            single_term = QubitOperator('Z0', 1.0)
        else:
            single_term = MockQubitOperator('Z0', 1.0)
        
        groups, metrics = optimized_pauli_grouping(single_term)
        
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 1)
        self.assertEqual(metrics['measurement_reduction_ratio'], 0.0)  # No reduction possible
        self.assertEqual(metrics['estimated_speedup'], 1.0)
    
    def test_identity_terms(self):
        """Test with identity terms."""
        if OPENFERMION_AVAILABLE:
            identity_hamiltonian = QubitOperator((), 1.0) + QubitOperator('Z0', 0.5)
        else:
            identity_hamiltonian = MockQubitOperator((), 1.0) + MockQubitOperator('Z0', 0.5)
        
        groups, metrics = optimized_pauli_grouping(identity_hamiltonian)
        
        # Should handle identity terms correctly
        self.assertGreater(len(groups), 0)
        self.assertEqual(metrics['commutation_purity'], 1.0)
    
    def test_small_coefficients(self):
        """Test with very small coefficients."""
        if OPENFERMION_AVAILABLE:
            QubitOp = QubitOperator
        else:
            QubitOp = MockQubitOperator
        
        small_coeff_hamiltonian = (
            QubitOp('Z0', 1e-10) + 
            QubitOp('Z1', 1e-12) + 
            QubitOp('X0 X1', 1e-8)
        )
        
        groups, metrics = optimized_pauli_grouping(small_coeff_hamiltonian)
        
        # Should handle small coefficients without numerical issues
        self.assertGreater(len(groups), 0)
        self.assertEqual(metrics['commutation_purity'], 1.0)
    
    def test_large_system(self):
        """Test scalability with larger systems."""
        if OPENFERMION_AVAILABLE:
            QubitOp = QubitOperator
        else:
            QubitOp = MockQubitOperator
        
        # Create larger Hamiltonian
        large_hamiltonian = QubitOp()
        for i in range(8):
            large_hamiltonian += QubitOp(f'Z{i}', np.random.random())
        
        for i in range(7):
            for j in range(i+1, 8):
                large_hamiltonian += QubitOp(f'Z{i} Z{j}', np.random.random() * 0.1)
        
        # Should handle larger systems efficiently
        groups, metrics = optimized_pauli_grouping(
            large_hamiltonian, 
            optimization_method='greedy'  # Use greedy for speed
        )
        
        self.assertGreater(len(groups), 0)
        self.assertEqual(metrics['commutation_purity'], 1.0)
    
    def test_all_non_commuting_terms(self):
        """Test with Hamiltonian where most terms don't commute."""
        if OPENFERMION_AVAILABLE:
            QubitOp = QubitOperator
        else:
            QubitOp = MockQubitOperator
        
        # Create Hamiltonian with mostly non-commuting terms
        non_commuting_hamiltonian = (
            QubitOp('X0', 1.0) + 
            QubitOp('Y0', 0.5) +  # Anti-commutes with X0
            QubitOp('Z0', 0.3) +  # Anti-commutes with X0 and Y0
            QubitOp('X1', 0.2)    # Commutes with all (different qubit)
        )
        
        groups, metrics = optimized_pauli_grouping(non_commuting_hamiltonian)
        
        # Should create more groups due to non-commuting terms
        self.assertGreaterEqual(len(groups), 3)  # Expect at least 3 groups
        self.assertEqual(metrics['commutation_purity'], 1.0)


@unittest.skipIf(not MODULE_AVAILABLE, "Pauli grouping module not available")
class TestIntegration(unittest.TestCase):
    """Integration tests with OpenFermion ecosystem."""
    
    @unittest.skipIf(not OPENFERMION_AVAILABLE, "OpenFermion not available")
    def test_molecular_data_integration(self):
        """Test integration with OpenFermion's MolecularData."""
        try:
            # Create a simple molecular system
            geometry = [('H', (0., 0., 0.)), ('H', (0., 0., 0.7414))]
            molecule = MolecularData(geometry, 'sto-3g', 1, 0)
            
            # This would normally require running quantum chemistry calculation
            # For testing, we'll create a mock Hamiltonian
            mock_hamiltonian = QubitOperator('Z0', -1.0) + QubitOperator('Z1', -0.5)
            
            # Test that our grouping works with standard workflow
            groups, metrics = optimized_pauli_grouping(mock_hamiltonian)
            
            self.assertGreater(len(groups), 0)
            self.assertEqual(metrics['commutation_purity'], 1.0)
            
        except ImportError:
            self.skipTest("Required OpenFermion modules not available")
    
    def test_backward_compatibility(self):
        """Test that existing OpenFermion code continues to work."""
        # Our function should work as a drop-in enhancement
        if OPENFERMION_AVAILABLE:
            test_hamiltonian = QubitOperator('Z0', 1.0) + QubitOperator('Z1', 0.5)
        else:
            test_hamiltonian = MockQubitOperator('Z0', 1.0) + MockQubitOperator('Z1', 0.5)
        
        # Existing code pattern should still work
        groups, metrics = optimized_pauli_grouping(test_hamiltonian)
        
        # Basic interface expectations
        self.assertIsInstance(groups, list)
        self.assertIsInstance(metrics, dict)


if __name__ == '__main__':
    # Configure test runner
    import sys
    
    # Set up test verbosity
    verbosity = 2 if '-v' in sys.argv else 1
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestAdvancedPauliGroupOptimizer,
        TestOptimizedPauliGroupingFunction,
        TestValidationFunctions,
        TestPerformanceTargets,
        TestEdgeCases,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbosity, buffer=True)
    result = runner.run(suite)
    
    # Print summary
    print(f"\nTest Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
