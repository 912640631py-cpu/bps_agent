"""Backward-compatible module entry point for the PDF export worker."""

from bps_agent.pdf_export import PdfExportJob, main, run_pdf_job

__all__ = ["PdfExportJob", "main", "run_pdf_job"]


if __name__ == "__main__":
    raise SystemExit(main())
