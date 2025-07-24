#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Entry point for the spectral processing pipeline."""
from spectral_pipeline.cli import main

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', nargs='?', default='.')
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()
    main(args.data_dir, do_plot=not args.no_plot)
