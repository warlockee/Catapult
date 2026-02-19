import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion';

export function Help() {
    return (
        <div className="space-y-6">
            <div>
                <h2 className="text-2xl font-bold tracking-tight">Help & Documentation</h2>
                <p className="text-muted-foreground">Find answers to common questions and learn how to use the Model Registry.</p>
            </div>

            <Card>
                <CardHeader>
                    <CardTitle>Frequently Asked Questions</CardTitle>
                    <CardDescription>Common questions about managing models and deployments.</CardDescription>
                </CardHeader>
                <CardContent>
                    <Accordion type="single" collapsible className="w-full">
                        <AccordionItem value="item-1">
                            <AccordionTrigger>How do I register a new model?</AccordionTrigger>
                            <AccordionContent>
                                To register a new model, navigate to the "Models" page and click the "Register Model" button. You'll need to provide a name, description, and initial version details.
                            </AccordionContent>
                        </AccordionItem>
                        <AccordionItem value="item-2">
                            <AccordionTrigger>How do I create a new release?</AccordionTrigger>
                            <AccordionContent>
                                Go to the "Releases" page or a specific model's page. Click "Create Release", select the model version, and provide release notes.
                            </AccordionContent>
                        </AccordionItem>
                        <AccordionItem value="item-3">
                            <AccordionTrigger>How do I deploy a model?</AccordionTrigger>
                            <AccordionContent>
                                Navigate to the "Deployments" page. Click "New Deployment", select the model release you want to deploy, and choose the target environment (e.g., Staging, Production).
                            </AccordionContent>
                        </AccordionItem>
                    </Accordion>
                </CardContent>
            </Card>

            <Card>
                <CardHeader>
                    <CardTitle>Contact Support</CardTitle>
                    <CardDescription>Need more help? Reach out to our support team.</CardDescription>
                </CardHeader>
                <CardContent>
                    <p className="text-sm text-gray-600">
                        Issues: <a href="https://github.com/warlockee/Catapult/issues" className="text-blue-600 hover:underline">https://github.com/warlockee/Catapult/issues</a><br />
                        Documentation: <a href="https://github.com/warlockee/Catapult/wiki" className="text-blue-600 hover:underline">https://github.com/warlockee/Catapult/wiki</a>
                    </p>
                </CardContent>
            </Card>
        </div>
    );
}
