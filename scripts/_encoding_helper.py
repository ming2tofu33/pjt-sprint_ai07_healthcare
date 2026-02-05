#!/usr/bin/env python3
"""
Windows 인코딩 헬퍼
모든 scripts에서 import하여 사용
"""

import sys
import codecs

def fix_windows_encoding():
    """Windows에서 UTF-8 출력 문제 해결"""
    if sys.platform == 'win32':
        # stdout, stderr를 UTF-8로 강제 설정
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
