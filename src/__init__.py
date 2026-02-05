"""
Healthcare AI Project - Source Modules

Core modules for YOLO-based medical image object detection pipeline.
"""

# Version info
__version__ = "1.0.0"
__author__ = "Team #4 - AI07 Healthcare"

# Import main utilities
from .utils import (
    # Path Management
    setup_project_paths,
    count_files,
    
    # Dependency & Environment Checks
    check_dependencies,
    ensure_dependencies,
    check_data_exists,
    
    # Reproducibility
    set_seed,
    get_package_version,
    
    # Config Management
    load_config,
    save_config,
    load_yaml_with_inheritance,
    deep_merge_dict,
    get_default_config,
    get_project_defaults,
    
    # Logging & Experiment Tracking
    save_json,
    load_json,
    append_jsonl,
    run_command,
    get_git_info,
    create_run_manifest,
    init_experiment_registry,
    register_experiment,
    
    # Results Recording
    flatten_dict,
    safe_scalar,
    init_results_table,
    record_result,
    
    # Helper Functions
    print_section,
    print_dict,
)

__all__ = [
    # Path Management
    "setup_project_paths",
    "count_files",
    
    # Dependency & Environment Checks
    "check_dependencies",
    "ensure_dependencies",
    "check_data_exists",
    
    # Reproducibility
    "set_seed",
    "get_package_version",
    
    # Config Management
    "load_config",
    "save_config",
    "load_yaml_with_inheritance",
    "deep_merge_dict",
    "get_default_config",
    "get_project_defaults",
    
    # Logging & Experiment Tracking
    "save_json",
    "load_json",
    "append_jsonl",
    "run_command",
    "get_git_info",
    "create_run_manifest",
    "init_experiment_registry",
    "register_experiment",
    
    # Results Recording
    "flatten_dict",
    "safe_scalar",
    "init_results_table",
    "record_result",
    
    # Helper Functions
    "print_section",
    "print_dict",
]
