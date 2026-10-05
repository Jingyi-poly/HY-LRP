"""Original report command and import entry point for the current LRP results.

Usage: python src-ini/report_generator.py --result RESULT.pkl --out REPORT.txt
"""
from setup.result_report import generate_report, generate_report_from_result, main

__all__ = ['generate_report', 'generate_report_from_result', 'main']


if __name__ == '__main__':
    main()
