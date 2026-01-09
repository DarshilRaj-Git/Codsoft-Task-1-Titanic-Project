#!/usr/bin/env python
"""
Script to run the professional Titanic ML project
"""

def main():
    print("Running Professional Titanic Survival Prediction ML Pipeline...")
    print("=" * 60)
    
    try:
        # Import and run the main pipeline
        from titanic_ml_project import main as run_pipeline
        results, best_model_name, encoders = run_pipeline()
        
        print("\n" + "=" * 60)
        print("EXECUTION SUMMARY")
        print("=" * 60)
        print(f"Best performing model: {best_model_name}")
        
        print("\nPipeline completed successfully!")
        print("\nThe project includes:")
        print("- Complete EDA with visualizations")
        print("- Data preprocessing and feature engineering")
        print("- Multiple model training and evaluation")
        print("- Professional ML workflow implementation")
        print("- Custom prediction functionality")
        
    except ImportError as e:
        print(f"Error importing module: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install -r requirements.txt")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()