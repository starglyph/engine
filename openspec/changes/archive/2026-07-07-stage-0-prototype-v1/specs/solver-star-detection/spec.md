## MODIFIED Requirements

### Requirement: Detection quality is measurable against truth
The system MUST compute precision and recall for detected stars against reference truth using configurable matching tolerance.

#### Scenario: 12 MP detection completes within budget
- **WHEN** a 12 MP synthetic star field is detected on a release build
- **THEN** wall-clock detection time is at most 1.5 seconds
