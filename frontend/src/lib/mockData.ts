export interface Image {
  id: string;
  name: string;
  repository: string;
  description: string;
  created_at: string;
  updated_at: string;
  release_count: number;
  latest_version: string;
}

export interface Release {
  id: string;
  image_id: string;
  image_name: string;
  version: string;
  tag: string;
  digest: string;
  size_bytes: number;
  platform: string;
  architecture: string;
  os: string;
  status: 'active' | 'deprecated' | 'archived';
  created_at: string;
  metadata: {
    git_commit?: string;
    git_branch?: string;
    build_timestamp?: string;
    ci_pipeline?: string;
    maintainer?: string;
    model_architecture?: string;
    pytorch_version?: string;
    cuda_version?: string;
    training_dataset?: string;
    accuracy?: number;
    loss?: number;
    epochs?: number;
    training_duration?: string;
  };
  ceph_path?: string;
}

export interface Deployment {
  id: string;
  release_id: string;
  release_version: string;
  image_name: string;
  environment: string;
  deployed_by: string;
  deployed_at: string;
  status: 'success' | 'failed' | 'in_progress';
  metadata: {
    kubernetes_namespace?: string;
    replicas?: number;
    gpu_type?: string;
  };
}

export interface ApiKey {
  id: string;
  name: string;
  key: string;
  is_active: boolean;
  created_at: string;
  last_used_at: string;
  expires_at?: string;
}

export const mockImages: Image[] = [
  {
    id: '1',
    name: 'pytorch-resnet50',
    repository: 'ml-team/pytorch-resnet50',
    description: 'ResNet-50 model for image classification on ImageNet',
    created_at: '2024-01-15T10:30:00Z',
    updated_at: '2024-11-18T14:20:00Z',
    release_count: 24,
    latest_version: '2.3.1',
  },
  {
    id: '2',
    name: 'bert-sentiment',
    repository: 'ml-team/bert-sentiment',
    description: 'BERT-base fine-tuned for sentiment analysis',
    created_at: '2024-03-22T08:15:00Z',
    updated_at: '2024-11-17T16:45:00Z',
    release_count: 18,
    latest_version: '1.5.2',
  },
  {
    id: '3',
    name: 'yolov8-detection',
    repository: 'ml-team/yolov8-detection',
    description: 'YOLOv8 object detection model for autonomous driving',
    created_at: '2024-05-10T12:00:00Z',
    updated_at: '2024-11-18T09:30:00Z',
    release_count: 31,
    latest_version: '3.1.0',
  },
  {
    id: '4',
    name: 'gpt-finetuned',
    repository: 'ml-team/gpt-finetuned',
    description: 'GPT-2 fine-tuned on domain-specific text generation',
    created_at: '2024-07-08T14:30:00Z',
    updated_at: '2024-11-16T11:20:00Z',
    release_count: 12,
    latest_version: '1.2.0',
  },
  {
    id: '5',
    name: 'efficientnet-mobile',
    repository: 'ml-team/efficientnet-mobile',
    description: 'EfficientNet optimized for mobile deployment',
    created_at: '2024-08-20T09:45:00Z',
    updated_at: '2024-11-15T13:10:00Z',
    release_count: 15,
    latest_version: '1.8.3',
  },
  {
    id: '6',
    name: 'stable-diffusion-v2',
    repository: 'ml-team/stable-diffusion-v2',
    description: 'Stable Diffusion v2 for image generation',
    created_at: '2024-09-12T16:20:00Z',
    updated_at: '2024-11-14T10:05:00Z',
    release_count: 8,
    latest_version: '2.0.1',
  },
];

export const mockReleases: Release[] = [
  {
    id: 'r1',
    image_id: '1',
    image_name: 'pytorch-resnet50',
    version: '2.3.1',
    tag: 'v2.3.1',
    digest: 'sha256:a8f9c3e2d1b4567890abcdef1234567890abcdef1234567890abcdef12345678',
    size_bytes: 524288000,
    platform: 'linux/amd64',
    architecture: 'amd64',
    os: 'linux',
    status: 'active',
    created_at: '2024-11-18T14:20:00Z',
    metadata: {
      git_commit: 'a7f9e3c',
      git_branch: 'main',
      build_timestamp: '2024-11-18T14:15:00Z',
      ci_pipeline: 'github-actions-1234',
      maintainer: 'ml-team@example.com',
      model_architecture: 'ResNet50',
      pytorch_version: '2.1.0',
      cuda_version: '12.1',
      training_dataset: 'ImageNet-1K',
      accuracy: 0.956,
      loss: 0.123,
      epochs: 100,
      training_duration: '4h 32m',
    },
    ceph_path: '/models/pytorch-resnet50/v2.3.1',
  },
  {
    id: 'r2',
    image_id: '2',
    image_name: 'bert-sentiment',
    version: '1.5.2',
    tag: 'v1.5.2',
    digest: 'sha256:b9e0d4f3e2c5678901bcdef2345678901bcdef2345678901bcdef23456789012',
    size_bytes: 438000000,
    platform: 'linux/amd64',
    architecture: 'amd64',
    os: 'linux',
    status: 'active',
    created_at: '2024-11-17T16:45:00Z',
    metadata: {
      git_commit: 'b8e0d4f',
      git_branch: 'main',
      build_timestamp: '2024-11-17T16:40:00Z',
      model_architecture: 'BERT-base',
      pytorch_version: '2.0.1',
      cuda_version: '11.8',
      training_dataset: 'IMDB-50K',
      accuracy: 0.923,
      loss: 0.234,
      epochs: 50,
      training_duration: '2h 15m',
    },
    ceph_path: '/models/bert-sentiment/v1.5.2',
  },
  {
    id: 'r3',
    image_id: '3',
    image_name: 'yolov8-detection',
    version: '3.1.0',
    tag: 'v3.1.0',
    digest: 'sha256:c0f1e5g4h3d6789012cdef3456789012cdef3456789012cdef34567890123456',
    size_bytes: 892000000,
    platform: 'linux/amd64',
    architecture: 'amd64',
    os: 'linux',
    status: 'active',
    created_at: '2024-11-18T09:30:00Z',
    metadata: {
      git_commit: 'c9f1e5g',
      git_branch: 'release/3.1',
      build_timestamp: '2024-11-18T09:25:00Z',
      model_architecture: 'YOLOv8',
      pytorch_version: '2.1.1',
      cuda_version: '12.2',
      training_dataset: 'COCO-2017 + Custom',
      accuracy: 0.889,
      loss: 0.456,
      epochs: 300,
      training_duration: '12h 45m',
    },
    ceph_path: '/models/yolov8-detection/v3.1.0',
  },
];

export const mockDeployments: Deployment[] = [
  {
    id: 'd1',
    release_id: 'r1',
    release_version: '2.3.1',
    image_name: 'pytorch-resnet50',
    environment: 'production',
    deployed_by: 'admin-key',
    deployed_at: '2024-11-18T15:00:00Z',
    status: 'success',
    metadata: {
      kubernetes_namespace: 'ml-prod',
      replicas: 3,
      gpu_type: 'A100',
    },
  },
  {
    id: 'd2',
    release_id: 'r2',
    release_version: '1.5.2',
    image_name: 'bert-sentiment',
    environment: 'staging',
    deployed_by: 'ci-cd-key',
    deployed_at: '2024-11-17T17:00:00Z',
    status: 'success',
    metadata: {
      kubernetes_namespace: 'ml-staging',
      replicas: 2,
      gpu_type: 'V100',
    },
  },
  {
    id: 'd3',
    release_id: 'r3',
    release_version: '3.1.0',
    image_name: 'yolov8-detection',
    environment: 'production',
    deployed_by: 'admin-key',
    deployed_at: '2024-11-18T10:00:00Z',
    status: 'success',
    metadata: {
      kubernetes_namespace: 'ml-prod',
      replicas: 5,
      gpu_type: 'A100',
    },
  },
  {
    id: 'd4',
    release_id: 'r1',
    release_version: '2.3.0',
    image_name: 'pytorch-resnet50',
    environment: 'production',
    deployed_by: 'admin-key',
    deployed_at: '2024-11-15T14:30:00Z',
    status: 'success',
    metadata: {
      kubernetes_namespace: 'ml-prod',
      replicas: 3,
      gpu_type: 'A100',
    },
  },
  {
    id: 'd5',
    release_id: 'r2',
    release_version: '1.5.1',
    image_name: 'bert-sentiment',
    environment: 'production',
    deployed_by: 'ci-cd-key',
    deployed_at: '2024-11-14T16:20:00Z',
    status: 'success',
    metadata: {
      kubernetes_namespace: 'ml-prod',
      replicas: 2,
      gpu_type: 'V100',
    },
  },
];

export const mockApiKeys: ApiKey[] = [
  {
    id: 'k1',
    name: 'admin-key',
    key: 'mk_live_abc123def456ghi789jkl012mno345pqr678',
    is_active: true,
    created_at: '2024-01-10T08:00:00Z',
    last_used_at: '2024-11-18T15:00:00Z',
  },
  {
    id: 'k2',
    name: 'ci-cd-key',
    key: 'mk_live_stu901vwx234yz567abc890def123ghi456jkl',
    is_active: true,
    created_at: '2024-02-15T10:30:00Z',
    last_used_at: '2024-11-17T17:00:00Z',
  },
  {
    id: 'k3',
    name: 'training-pipeline',
    key: 'mk_live_mno789pqr012stu345vwx678yz901abc234def',
    is_active: true,
    created_at: '2024-06-20T14:15:00Z',
    last_used_at: '2024-11-16T12:45:00Z',
  },
  {
    id: 'k4',
    name: 'legacy-key',
    key: 'mk_live_ghi567jkl890mno123pqr456stu789vwx012yz',
    is_active: false,
    created_at: '2023-11-05T09:00:00Z',
    last_used_at: '2024-08-22T11:30:00Z',
    expires_at: '2024-09-01T00:00:00Z',
  },
];
