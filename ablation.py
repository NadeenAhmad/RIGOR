from owlready2 import get_ontology
import pandas as pd

class OWLOntologyEvaluator:
    def __init__(self, gold_standard_path, ablation_study_path):
        self.gold_standard_path = gold_standard_path
        self.ablation_study_path = ablation_study_path
        self.gold_standard = None
        self.ablation_study = None
    
    def load_ontologies(self):
        """Load both OWL ontologies"""
        print("Loading ontologies...")
        self.gold_standard = get_ontology(self.gold_standard_path).load()
        self.ablation_study = get_ontology(self.ablation_study_path).load()
        print("Ontologies loaded successfully!")
    
    def extract_classes(self, ontology):
        """Extract all classes from ontology"""
        return set(ontology.classes())
    
    def extract_data_properties(self, ontology):
        """Extract all data properties from ontology"""
        return set(ontology.data_properties())
    
    def extract_object_properties(self, ontology):
        """Extract all object properties from ontology"""
        return set(ontology.object_properties())
    
    def extract_annotation_properties(self, ontology):
        """Extract all annotation properties from ontology"""
        return set(ontology.annotation_properties())
    
    def get_element_name(self, element):
        """Get readable name for ontology element"""
        return str(element).split('.')[-1] if hasattr(element, 'name') else str(element)
    
    def calculate_metrics(self, gold_set, ablation_set):
        """Calculate precision, recall, and F1 score"""
        gold = set(gold_set)
        ablation = set(ablation_set)
        
        tp = len(gold.intersection(ablation))
        fp = len(ablation - gold)
        fn = len(gold - ablation)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4)
        }
    
    def detailed_comparison(self, gold_set, ablation_set, category_name):
        """Provide detailed element-level comparison"""
        gold = set(gold_set)
        ablation = set(ablation_set)
        
        tp_elements = gold.intersection(ablation)
        fp_elements = ablation - gold
        fn_elements = gold - ablation
        
        print(f"\n{'='*60}")
        print(f"DETAILED {category_name.upper()} ANALYSIS")
        print(f"{'='*60}")
        print(f"True Positives ({len(tp_elements)}):")
        for element in sorted(tp_elements, key=self.get_element_name):
            print(f"  ✓ {self.get_element_name(element)}")
        
        print(f"\nFalse Positives ({len(fp_elements)}):")
        for element in sorted(fp_elements, key=self.get_element_name):
            print(f"  ✗ {self.get_element_name(element)}")
        
        print(f"\nFalse Negatives ({len(fn_elements)}):")
        for element in sorted(fn_elements, key=self.get_element_name):
            print(f"  ✗ {self.get_element_name(element)}")
        
        return tp_elements, fp_elements, fn_elements
    
    def evaluate_classes(self, verbose=True):
        """Evaluate class-level metrics"""
        gold_classes = self.extract_classes(self.gold_standard)
        ablation_classes = self.extract_classes(self.ablation_study)
        
        if verbose:
            self.detailed_comparison(gold_classes, ablation_classes, "CLASSES")
        
        return self.calculate_metrics(gold_classes, ablation_classes)
    
    def evaluate_data_properties(self, verbose=True):
        """Evaluate data property metrics"""
        gold_dp = self.extract_data_properties(self.gold_standard)
        ablation_dp = self.extract_data_properties(self.ablation_study)
        
        if verbose:
            self.detailed_comparison(gold_dp, ablation_dp, "DATA PROPERTIES")
        
        return self.calculate_metrics(gold_dp, ablation_dp)
    
    def evaluate_object_properties(self, verbose=True):
        """Evaluate object property metrics"""
        gold_op = self.extract_object_properties(self.gold_standard)
        ablation_op = self.extract_object_properties(self.ablation_study)
        
        if verbose:
            self.detailed_comparison(gold_op, ablation_op, "OBJECT PROPERTIES")
        
        return self.calculate_metrics(gold_op, ablation_op)
    
    def evaluate_annotation_properties(self, verbose=True):
        """Evaluate annotation property metrics"""
        gold_ap = self.extract_annotation_properties(self.gold_standard)
        ablation_ap = self.extract_annotation_properties(self.ablation_study)
        
        if verbose:
            self.detailed_comparison(gold_ap, ablation_ap, "ANNOTATION PROPERTIES")
        
        return self.calculate_metrics(gold_ap, ablation_ap)
    
    def get_ontology_stats(self, ontology, name):
        """Get basic statistics for an ontology"""
        stats = {
            'name': name,
            'classes': len(self.extract_classes(ontology)),
            'data_properties': len(self.extract_data_properties(ontology)),
            'object_properties': len(self.extract_object_properties(ontology)),
            'annotation_properties': len(self.extract_annotation_properties(ontology))
        }
        return stats
    
    def comprehensive_evaluation(self, verbose=True):
        """Run complete evaluation"""
        print("Starting comprehensive ontology evaluation...")
        
        # Load ontologies
        self.load_ontologies()
        
        # Print basic statistics
        print("\nONTOLOGY STATISTICS:")
        print("-" * 50)
        gold_stats = self.get_ontology_stats(self.gold_standard, "Gold Standard")
        ablation_stats = self.get_ontology_stats(self.ablation_study, "Ablation Study")
        
        stats_df = pd.DataFrame([gold_stats, ablation_stats])
        print(stats_df.to_string(index=False))
        
        # Run evaluations
        results = {
            'classes': self.evaluate_classes(verbose),
            'data_properties': self.evaluate_data_properties(verbose),
            'object_properties': self.evaluate_object_properties(verbose),
            'annotation_properties': self.evaluate_annotation_properties(verbose)
        }
        
        # Calculate overall metrics (micro-average)
        total_tp = sum(results[category]['true_positives'] for category in results)
        total_fp = sum(results[category]['false_positives'] for category in results)
        total_fn = sum(results[category]['false_negatives'] for category in results)
        
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        
        results['overall'] = {
            'true_positives': total_tp,
            'false_positives': total_fp,
            'false_negatives': total_fn,
            'precision': round(overall_precision, 4),
            'recall': round(overall_recall, 4),
            'f1_score': round(overall_f1, 4)
        }
        
        # Print summary results
        self.print_results_summary(results)
        
        return results
    
    def print_results_summary(self, results):
        """Print formatted results summary"""
        print(f"\n{'='*80}")
        print("EVALUATION RESULTS SUMMARY")
        print(f"{'='*80}")
        
        headers = ["Category", "Precision", "Recall", "F1-Score", "TP", "FP", "FN"]
        print(f"{headers[0]:<20} {headers[1]:<10} {headers[2]:<10} {headers[3]:<10} {headers[4]:<5} {headers[5]:<5} {headers[6]:<5}")
        print("-" * 80)
        
        for category, metrics in results.items():
            if category != 'overall':
                print(f"{category:<20} {metrics['precision']:<10} {metrics['recall']:<10} {metrics['f1_score']:<10} "
                      f"{metrics['true_positives']:<5} {metrics['false_positives']:<5} {metrics['false_negatives']:<5}")
        
        print("-" * 80)
        overall = results['overall']
        print(f"{'OVERALL':<20} {overall['precision']:<10} {overall['recall']:<10} {overall['f1_score']:<10} "
              f"{overall['true_positives']:<5} {overall['false_positives']:<5} {overall['false_negatives']:<5}")
        print(f"{'='*80}")

if __name__ == "__main__":
    gold_standard_path = "path/to/*Ontology"
    ablation_study_path = "path/to/rag_ontology"
    
    evaluator = OWLOntologyEvaluator(gold_standard_path, ablation_study_path)
    
    results = evaluator.comprehensive_evaluation(verbose=True)
    
    results_df = pd.DataFrame(results).T
    results_df.to_csv("ontology_evaluation_results.csv")
    print("\nResults saved to 'ontology_evaluation_results_wo_relevant_doc.csv'")