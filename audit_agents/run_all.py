"""
🚀 Run All Audit Agents
========================
Lancia tutti gli 8 agenti audit in parallelo usando MCP Orchestrator.
"""

import asyncio
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from audit_agents.orchestrator import MCPOrchestrator
from audit_agents.data_flow_auditor import DataFlowAuditor


# Import other auditors (to be implemented)
# from audit_agents.investigation_auditor import InvestigationAuditor
# from audit_agents.knowledge_auditor import KnowledgeAuditor
# from audit_agents.api_auditor import APIAuditor
# from audit_agents.causal_auditor import CausalAuditor
# from audit_agents.quality_auditor import QualityAuditor
# from audit_agents.frontend_auditor import FrontendAuditor
# from audit_agents.ops_auditor import OpsAuditor


async def main():
    """Run full multi-agent audit."""
    print("🤖 Multi-Agent Audit System")
    print("=" * 60)
    print("Version: 1.0.0")
    print("Date: 2025-12-25")
    print("")
    
    # Create orchestrator
    orchestrator = MCPOrchestrator(output_dir=Path("audit_reports"))
    
    # Register all agents
    print("📋 Registering agents...")
    orchestrator.register_agent(DataFlowAuditor())
    
    # TODO: Uncomment as agents are implemented
    # orchestrator.register_agent(InvestigationAuditor())
    # orchestrator.register_agent(KnowledgeAuditor())
    # orchestrator.register_agent(APIAuditor())
    # orchestrator.register_agent(CausalAuditor())
    # orchestrator.register_agent(QualityAuditor())
    # orchestrator.register_agent(FrontendAuditor())
    # orchestrator.register_agent(OpsAuditor())
    
    print("")
    
    # Run parallel audit
    report = await orchestrator.run_parallel()
    
    # Display summary
    print("\n" + "=" * 60)
    print("📊 AUDIT SUMMARY")
    print("=" * 60)
    summary = report['summary']
    print(f"Total Checks: {summary['total_checks']}")
    print(f"Passed:       {summary['passed']} ✅")
    print(f"Failed:       {summary['failed']} ❌")
    print(f"Warnings:     {summary['warnings']} ⚠️")
    print(f"Pass Rate:    {summary['pass_rate']}")
    print("")
    
    # Critical issues
    if report['critical_issues']:
        print("🔴 CRITICAL ISSUES")
        print("-" * 60)
        for issue in report['critical_issues'][:5]:  # Show top 5
            print(f"  • [{issue['agent']}] {issue['message']}")
        if len(report['critical_issues']) > 5:
            print(f"  ... and {len(report['critical_issues']) - 5} more")
        print("")
    
    # Recommendations
    if report['recommendations']:
        print("💡 TOP RECOMMENDATIONS")
        print("-" * 60)
        for rec in report['recommendations'][:3]:  # Show top 3
            print(f"  {rec['priority'].upper()}: {rec['area']}")
            print(f"    {rec['issue']}")
            print(f"    Effort: {rec['effort']}")
            print("")
    
    # Save reports
    print("💾 Saving reports...")
    orchestrator.save_report(report, "audit_report.json")
    orchestrator.save_markdown(report, "AUDIT_REPORT.md")
    
    print("\n✅ Audit complete!")
    return report


if __name__ == "__main__":
    asyncio.run(main())
