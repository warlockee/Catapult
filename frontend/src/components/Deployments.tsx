import { DeploymentHistory } from './DeploymentHistory';

export function Deployments() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Deployments</h1>
        <p className="text-gray-500 mt-1">Manage dev deployments</p>
      </div>

      <DeploymentHistory embedded />
    </div>
  );
}
