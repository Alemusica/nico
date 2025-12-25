"""
🎭 MCP Orchestrator - Multi-Agent Coordination
==============================================
Coordina esecuzione parallela di 8 agenti audit specializzati.
"""

import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json


@dataclass
class AuditCheck:
    """Single audit check result."""
    name: str
    passed: bool
    severity: str  # "critical", "high", "medium", "low"
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0


@dataclass
class AgentReport:
    """Report from single audit agent."""
    agent_name: str
    scope: str
    total_checks: int
    passed: int
    failed: int
    warnings: int
    checks: List[AuditCheck]
    duration_ms: float
    error: Optional[str] = None


class AuditAgent:
    """Base class for audit agents."""
    
    def __init__(self, name: str, scope: str):
        self.name = name
        self.scope = scope
        self.checks: List[AuditCheck] = []
    
    async def run_audit(self) -> AgentReport:
        """
        Run audit checks and return report.
        Override in subclasses.
        """
        start_time = datetime.now()
        
        try:
            # Run checks (override this)
            await self._run_checks()
            
            # Aggregate results
            passed = sum(1 for c in self.checks if c.passed)
            failed = sum(1 for c in self.checks if not c.passed and c.severity in ["critical", "high"])
            warnings = sum(1 for c in self.checks if not c.passed and c.severity in ["medium", "low"])
            
            duration = (datetime.now() - start_time).total_seconds() * 1000
            
            return AgentReport(
                agent_name=self.name,
                scope=self.scope,
                total_checks=len(self.checks),
                passed=passed,
                failed=failed,
                warnings=warnings,
                checks=self.checks,
                duration_ms=duration
            )
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds() * 1000
            return AgentReport(
                agent_name=self.name,
                scope=self.scope,
                total_checks=0,
                passed=0,
                failed=0,
                warnings=0,
                checks=[],
                duration_ms=duration,
                error=str(e)
            )
    
    async def _run_checks(self):
        """Override in subclass to implement checks."""
        pass
    
    def add_check(self, check: AuditCheck):
        """Add check result."""
        self.checks.append(check)
    
    def check(self, name: str, condition: bool, severity: str, message: str, **details):
        """Helper to add check result."""
        self.add_check(AuditCheck(
            name=name,
            passed=condition,
            severity=severity,
            message=message,
            details=details
        ))


class MCPOrchestrator:
    """
    Orchestrator for parallel agent execution.
    
    Usage:
        orchestrator = MCPOrchestrator()
        orchestrator.register_agent(DataFlowAuditor())
        orchestrator.register_agent(InvestigationAuditor())
        report = await orchestrator.run_parallel()
    """
    
    def __init__(self, output_dir: Path = Path("audit_reports")):
        self.agents: List[AuditAgent] = []
        self.reports: Dict[str, AgentReport] = {}
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
    
    def register_agent(self, agent: AuditAgent):
        """Register audit agent."""
        self.agents.append(agent)
        print(f"  ✅ Registered: {agent.name}")
    
    async def run_parallel(self) -> Dict[str, Any]:
        """Execute all agents in parallel."""
        print(f"\n🚀 Launching {len(self.agents)} audit agents in parallel...")
        start_time = datetime.now()
        
        # Run all agents concurrently
        tasks = [agent.run_audit() for agent in self.agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect results
        for agent, result in zip(self.agents, results):
            if isinstance(result, Exception):
                print(f"  ❌ {agent.name}: {result}")
                self.reports[agent.name] = AgentReport(
                    agent_name=agent.name,
                    scope=agent.scope,
                    total_checks=0,
                    passed=0,
                    failed=0,
                    warnings=0,
                    checks=[],
                    duration_ms=0,
                    error=str(result)
                )
            else:
                status = "✅" if result.failed == 0 else "⚠️"
                print(f"  {status} {agent.name}: {result.passed}/{result.total_checks} passed")
                self.reports[agent.name] = result
        
        duration = (datetime.now() - start_time).total_seconds()
        print(f"\n⏱️ Total audit time: {duration:.2f}s")
        
        # Generate aggregated report
        return self.aggregate_report()
    
    def aggregate_report(self) -> Dict[str, Any]:
        """Combine all agent reports into summary."""
        total_checks = sum(r.total_checks for r in self.reports.values())
        total_passed = sum(r.passed for r in self.reports.values())
        total_failed = sum(r.failed for r in self.reports.values())
        total_warnings = sum(r.warnings for r in self.reports.values())
        
        # Extract critical issues
        critical_issues = []
        for agent_name, report in self.reports.items():
            if report.error:
                critical_issues.append({
                    "agent": agent_name,
                    "type": "agent_error",
                    "message": report.error
                })
            else:
                for check in report.checks:
                    if not check.passed and check.severity == "critical":
                        critical_issues.append({
                            "agent": agent_name,
                            "check": check.name,
                            "message": check.message,
                            "details": check.details
                        })
        
        return {
            "summary": {
                "timestamp": datetime.now().isoformat(),
                "total_agents": len(self.agents),
                "total_checks": total_checks,
                "passed": total_passed,
                "failed": total_failed,
                "warnings": total_warnings,
                "pass_rate": f"{total_passed/total_checks*100:.1f}%" if total_checks > 0 else "N/A",
                "agents_with_errors": sum(1 for r in self.reports.values() if r.error)
            },
            "agents": {name: self._report_to_dict(report) for name, report in self.reports.items()},
            "critical_issues": critical_issues,
            "recommendations": self.generate_recommendations()
        }
    
    def _report_to_dict(self, report: AgentReport) -> Dict[str, Any]:
        """Convert report to dict."""
        return {
            "scope": report.scope,
            "total_checks": report.total_checks,
            "passed": report.passed,
            "failed": report.failed,
            "warnings": report.warnings,
            "duration_ms": report.duration_ms,
            "error": report.error,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "severity": c.severity,
                    "message": c.message,
                    "details": c.details
                }
                for c in report.checks
            ]
        }
    
    def generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate prioritized recommendations."""
        recommendations = []
        
        # Analyze patterns across agents
        for agent_name, report in self.reports.items():
            if report.error:
                recommendations.append({
                    "priority": "critical",
                    "area": agent_name,
                    "issue": f"Agent failed to execute: {report.error}",
                    "action": "Fix agent implementation and retry",
                    "effort": "medium"
                })
            else:
                failed_critical = [c for c in report.checks if not c.passed and c.severity == "critical"]
                if failed_critical:
                    recommendations.append({
                        "priority": "high",
                        "area": agent_name,
                        "issue": f"{len(failed_critical)} critical checks failed",
                        "action": "\n".join(f"- {c.name}: {c.message}" for c in failed_critical[:3]),
                        "effort": "high"
                    })
        
        # Sort by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        recommendations.sort(key=lambda x: priority_order.get(x["priority"], 99))
        
        return recommendations
    
    def save_report(self, report: Dict[str, Any], filename: str = "audit_report.json"):
        """Save report to file."""
        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n💾 Report saved: {output_path}")
        return output_path
    
    def generate_markdown(self, report: Dict[str, Any]) -> str:
        """Generate Markdown report."""
        md = []
        md.append("# 🔍 Multi-Agent Audit Report")
        md.append(f"\n**Generated**: {report['summary']['timestamp']}")
        md.append(f"**Agents**: {report['summary']['total_agents']}")
        md.append("")
        
        # Summary
        md.append("## 📊 Summary")
        md.append("")
        md.append(f"- **Total Checks**: {report['summary']['total_checks']}")
        md.append(f"- **Passed**: {report['summary']['passed']} ✅")
        md.append(f"- **Failed**: {report['summary']['failed']} ❌")
        md.append(f"- **Warnings**: {report['summary']['warnings']} ⚠️")
        md.append(f"- **Pass Rate**: {report['summary']['pass_rate']}")
        md.append("")
        
        # Critical Issues
        if report['critical_issues']:
            md.append("## 🔴 Critical Issues")
            md.append("")
            for i, issue in enumerate(report['critical_issues'], 1):
                md.append(f"### {i}. {issue['agent']}")
                if issue.get('check'):
                    md.append(f"**Check**: {issue['check']}")
                md.append(f"**Issue**: {issue['message']}")
                if issue.get('details'):
                    md.append(f"**Details**: {json.dumps(issue['details'], indent=2)}")
                md.append("")
        
        # Recommendations
        md.append("## 💡 Recommendations")
        md.append("")
        for rec in report['recommendations']:
            priority_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
            emoji = priority_emoji.get(rec['priority'], "⚪")
            md.append(f"### {emoji} {rec['area']} ({rec['priority'].upper()})")
            md.append(f"**Issue**: {rec['issue']}")
            md.append(f"**Action**:")
            md.append(rec['action'])
            md.append(f"**Effort**: {rec['effort']}")
            md.append("")
        
        # Agent Details
        md.append("## 🤖 Agent Reports")
        md.append("")
        for agent_name, agent_report in report['agents'].items():
            status = "✅" if agent_report['failed'] == 0 else "⚠️"
            md.append(f"### {status} {agent_name}")
            md.append(f"**Scope**: {agent_report['scope']}")
            md.append(f"**Checks**: {agent_report['passed']}/{agent_report['total_checks']} passed")
            md.append(f"**Duration**: {agent_report['duration_ms']:.0f}ms")
            if agent_report.get('error'):
                md.append(f"**Error**: {agent_report['error']}")
            md.append("")
        
        return "\n".join(md)
    
    def save_markdown(self, report: Dict[str, Any], filename: str = "AUDIT_REPORT.md"):
        """Save Markdown report."""
        md_content = self.generate_markdown(report)
        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            f.write(md_content)
        print(f"📝 Markdown report saved: {output_path}")
        return output_path


# Example usage
async def main():
    """Example orchestrator usage."""
    orchestrator = MCPOrchestrator()
    
    # Register agents (to be implemented)
    # orchestrator.register_agent(DataFlowAuditor())
    # orchestrator.register_agent(InvestigationAuditor())
    # ... register all 8 agents
    
    # Run parallel audit
    report = await orchestrator.run_parallel()
    
    # Save reports
    orchestrator.save_report(report)
    orchestrator.save_markdown(report)
    
    return report


if __name__ == "__main__":
    print("🤖 MCP Orchestrator Ready")
    print("=" * 60)
    # asyncio.run(main())
